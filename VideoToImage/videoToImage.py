import cv2
import os

def video_to_images(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算需要提取的帧数（每秒提取frame_interval帧）
    target_frame_count = 100
    step = max(1, int(fps * (total_frames / target_frame_count)))
    
    count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            img_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1
            if saved_count >= target_frame_count:
                break
        count += 1
    
    cap.release()
    print(f"成功提取{saved_count}张图片至{output_folder}")

def video_to_images(video_path, output_folder, frame_interval=1):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            img_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            saved_count += 1
        count += 1
    
    cap.release()
    print(f"成功提取{saved_count}张图片至{output_folder}")


# 使用示例
video_to_images("bobo.mp4", "output_images", frame_interval=5)