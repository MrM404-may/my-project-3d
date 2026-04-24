#
# Copyright (C) 2026 - Continuous Update Training Script
# 基于GaussianUpdate的持续更新训练脚本
#

import os
import sys
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, prefilter_voxel
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_continuous_update(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, 
                               checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, 
                               ply_path=None, num_timestamps=5, timestamp_duration=30000):
    """
    持续更新训练函数，处理多个时间戳的数据
    
    参数:
        num_timestamps: 时间戳数量
        timestamp_duration: 每个时间戳的训练迭代次数
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,
        num_regions=dataset.num_regions if hasattr(dataset, 'num_regions') else 65
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
    gaussians.training_setup(opt)
    gaussians.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)
    
    # ==============================
    # 初始化持续更新管理器
    # ==============================
    print("\n" + "=" * 60)
    print("初始化持续更新管理器...")
    gaussians.init_continuous_update(
        num_regions=dataset.num_regions if hasattr(dataset, 'num_regions') else 65
    )
    print("=" * 60 + "\n")
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # ==============================
    # 持续更新：遍历每个时间戳
    # ==============================
    for timestamp_idx in range(num_timestamps):
        # 归一化时间戳到 [0, 1]
        normalized_timestamp = timestamp_idx / max(1, num_timestamps - 1)
        
        print(f"\n{'='*60}")
        print(f"时间戳 {timestamp_idx + 1}/{num_timestamps} (归一化值: {normalized_timestamp:.4f})")
        print(f"{'='*60}\n")
        
        # 初始化当前时间戳的更新
        gaussians.init_timestamp_update(normalized_timestamp)
        
        # 生成该时间戳的训练迭代
        timestamp_start_iter = timestamp_idx * timestamp_duration
        timestamp_end_iter = (timestamp_idx + 1) * timestamp_duration
        
        progress_bar = tqdm(range(timestamp_start_iter, timestamp_end_iter), 
                           desc=f"训练时间戳 {timestamp_idx + 1}")
        
        for iteration in progress_bar:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier, use_continuous_update=True)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()
            
            gaussians.update_learning_rate(iteration)
            
            # 更新持续更新阶段
            gaussians.update_training_stage()

            if dataset.random_background:
                bg_color = [torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item()]
            elif dataset.white_background:
                bg_color = [1.0, 1.0, 1.0]
            else:
                bg_color = [0.0, 0.0, 0.0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            # Pick a random camera from current timestamp
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # 获取当前相机所在的区域
            # 这里需要根据实际数据结构确定camera_region
            camera_region = 0  # 默认区域
            if hasattr(viewpoint_cam, 'region_id'):
                camera_region = viewpoint_cam.region_id

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            gaussians.set_anchor_mask(viewpoint_cam.camera_center, iteration, viewpoint_cam.resolution_scale)
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
            # 使用持续更新渲染
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, 
                               camera_region=camera_region, use_continuous_update=True)
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = \
                render_pkg["render"], render_pkg["viewspace_point_tensor"], render_pkg["visibility_filter"], \
                render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            gt_image = viewpoint_cam.original_image.cuda()
            
            # ==============================
            # 布局不变掩码：计算损失
            # ==============================
            # 默认使用完整图像计算损失
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            
            # 如果是第一阶段（外观更新），可以使用布局不变掩码
            current_stage = gaussians.get_current_stage()
            if current_stage == 'appearance':
                # 可以在这里添加布局不变掩码的应用
                # 例如：
                # with torch.no_grad():
                #     old_render = render_pkg["render"]  # 旧渲染
                #     mask = gaussians.generate_layout_invariant_mask(old_render, gt_image, camera_region)
                #     mask = mask.unsqueeze(0)
                #     Ll1 = l1_loss(image * mask, gt_image * mask) / (mask.sum() + 1e-6)
                #     ssim_loss = (1.0 - ssim(image * mask, gt_image * mask))
                pass
            
            # 基础损失
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
            
            # ==============================
            # 添加移除因子正则化损失
            # ==============================
            reg_loss = gaussians.get_removal_regularization_loss()
            if reg_loss > 0:
                loss = loss + 0.001 * reg_loss  # 可调权重

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                
                # 更新进度条信息，显示当前阶段
                stage_info = f"阶段: {current_stage}" if current_stage else ""
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Timestamp": f"{timestamp_idx + 1}/{num_timestamps}",
                    **({} if not stage_info else {"Stage": stage_info})
                })
                
                if iteration % 10 == 0:
                    progress_bar.update(10)
                
                if iteration == timestamp_end_iter - 1:
                    # 保存当前时间戳的状态到可见性池
                    gaussians.save_current_state_to_visibility_pool(scene.getTrainCameras())
                    progress_bar.close()

                # Log and save
                training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, 
                              iter_start.elapsed_time(iter_end), testing_iterations, scene, render, 
                              (pipe, background), wandb, logger, use_continuous_update=True)
                
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    # 保存持续更新检查点
                    gaussians.save_continuous_update_checkpoint(scene.model_path + "/continuous_update_chkpnt" + str(iteration) + ".pth")
                
                if (iteration in saving_iterations):
                    logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

    # 训练完成
    print("\n" + "="*60)
    print("持续更新训练完成！")
    print("="*60)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, 
                   testing_iterations, scene: Scene, renderFunc, renderArgs, wandb=None, 
                   logger=None, use_continuous_update=False):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, 'train_total_loss': loss})
    
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    scene.gaussians.set_anchor_mask(viewpoint.camera_center, iteration, viewpoint.resolution_scale)
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    
                    # 获取相机区域
                    camera_region = 0
                    if hasattr(viewpoint, 'region_id'):
                        camera_region = viewpoint.region_id
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, 
                                                   visible_mask=voxel_visible_mask, camera_region=camera_region,
                                                   use_continuous_update=use_continuous_update)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name']+"_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name']+"_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                        
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name']+"_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name']+'/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name']+'/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss": l1_test, f"{config['name']}_PSNR": psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--ply_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default='default')
    
    # 持续更新相关参数
    parser.add_argument("--num_timestamps", type=int, default=5, help="时间戳数量")
    parser.add_argument("--timestamp_duration", type=int, default=30000, help="每个时间戳的训练迭代次数")
    parser.add_argument("--num_regions", type=int, default=65, help="区域数量")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 设置参数
    args.save_iterations = args.save_iterations if args.save_iterations else [30000]
    args.checkpoint_iterations = args.checkpoint_iterations if args.checkpoint_iterations else [30000]
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # 启动GUI服务器
    import gaussian_renderer.network_gui as network_gui
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 开始持续更新训练
    training_continuous_update(
        lp.extract(args), op.extract(args), pp.extract(args), args.dataset_name,
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, ply_path=args.ply_path,
        num_timestamps=args.num_timestamps, timestamp_duration=args.timestamp_duration
    )

    # All done
    print("\nTraining complete.")
