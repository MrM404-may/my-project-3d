import cv2
import numpy as np

class QRCodePoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, qr_size=0.1):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.qr_size = qr_size
        self.qr_detector = cv2.QRCodeDetector()
        
        self.object_points = np.array([
            [-qr_size/2, -qr_size/2, 0],
            [qr_size/2, -qr_size/2, 0],
            [qr_size/2, qr_size/2, 0],
            [-qr_size/2, qr_size/2, 0]
        ], dtype=np.float32)
    
    def detect_and_estimate_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data, bbox, _ = self.qr_detector.detectAndDecode(gray)
        
        if bbox is not None and len(bbox) == 4:
            image_points = np.array(bbox[0], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if success:
                return data, rvec, tvec, image_points
        
        return None, None, None, None
    
    def draw_axis(self, frame, rvec, tvec):
        axis_length = self.qr_size * 0.5
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])
        
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[1]), (0, 0, 255), 3)
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[2]), (0, 255, 0), 3)
        frame = cv2.line(frame, tuple(imgpts[0]), tuple(imgpts[3]), (255, 0, 0), 3)
        
        return frame
    
    def draw_bbox(self, frame, image_points):
        for i in range(4):
            frame = cv2.line(frame, tuple(image_points[i]), tuple(image_points[(i+1)%4]), (255, 0, 0), 2)
        
        return frame

def main():
    camera_matrix = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    qr_size = 0.15
    
    estimator = QRCodePoseEstimator(camera_matrix, dist_coeffs, qr_size)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        
        data, rvec, tvec, bbox = estimator.detect_and_estimate_pose(frame)
        
        if data is not None:
            frame = estimator.draw_bbox(frame, bbox)
            
            if rvec is not None and tvec is not None:
                frame = estimator.draw_axis(frame, rvec, tvec)
                
                tvec_meters = tvec.flatten()
                print(f"二维码内容: {data}")
                print(f"位置 (x, y, z): ({tvec_meters[0]:.3f}, {tvec_meters[1]:.3f}, {tvec_meters[2]:.3f}) 米")
                
                rmat, _ = cv2.Rodrigues(rvec)
                euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))[6]
                print(f"姿态 (rx, ry, rz): ({euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}) 度")
                print("-" * 40)
        
        cv2.imshow("QR Code Pose Estimation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()