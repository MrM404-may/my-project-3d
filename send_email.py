import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

def send_mail(content):
    """
    发送邮件函数
    :param content: 要发送的邮件正文内容
    """
    # -------------------------- 请在这里修改你的配置 --------------------------
    sender = "1738945526@qq.com"       # 发件人邮箱
    auth_code = "gbnrtaqytikjbbee"         # 邮箱授权码（不是登录密码）
    receiver = "1738945526@qq.com"  # 收件人邮箱
    smtp_server = "smtp.qq.com"      # SMTP服务器
    smtp_port = 465                  # 端口
    # -------------------------------------------------------------------------

    # 构造邮件
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['From'] = formataddr(('发件人', sender))
    msg['To'] = formataddr(('收件人', receiver))
    msg['Subject'] = "Python 自动发送邮件"  # 邮件标题

    try:
        # 发送
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, auth_code)
            server.sendmail(sender, [receiver], msg.as_string())
        print("✅ 邮件发送成功")
    except Exception as e:
        print(f"❌ 发送失败：{e}")


# ------------------- 使用示例 -------------------
if __name__ == '__main__':
    # 只需要传入你想发送的内容
    send_mail("这是自动发送的测试内容，你好呀！")