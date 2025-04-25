import cv2
import os

# def video_to_images(video_path, output_folder):
#     # 创建输出文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # 读取视频
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # 计算需要提取的帧数（每秒提取frame_interval帧）
#     target_frame_count = 100
#     step = max(1, int(fps * (total_frames / target_frame_count)))
    
#     count = 0
#     saved_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % step == 0:
#             img_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
#             cv2.imwrite(img_path, frame)
#             saved_count += 1
#             if saved_count >= target_frame_count:
#                 break
#         count += 1
    
#     cap.release()
#     print(f"成功提取{saved_count}张图片至{output_folder}")

# def video_to_images(video_path, output_folder, frame_interval=1):
#     # 创建输出文件夹
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # 读取视频
#     cap = cv2.VideoCapture(video_path)
#     prev_frame = None
#     frame_count = 0
#     count = 0
#     saved_count = 0

#     threshold = frame_interval # 视频抽帧运动阈值
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # if count % frame_interval == 0:
#         #     img_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
#         #     cv2.imwrite(img_path, frame)
#         #     saved_count += 1
#         # count += 1
#         if prev_frame is not None:
#             diff = cv2.absdiff(frame, prev_frame)
#             if cv2.countNonZero(diff) > threshold:
#                 img_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
#                 cv2.imwrite(img_path, frame)
#                 saved_count += 1
#         prev_frame = frame.copy()
#     cap.release()
#     print(f"成功提取{saved_count}张图片至{output_folder}")


# # 使用示例
# video_to_images("bobo.mp4", "output_images", frame_interval=5)




def extract_key_frames(video_path, output_dir, threshold=30):
    """
    功能：自动检测视频中画面变化的时刻，只保存关键帧
    参数：
        video_path: 视频文件路径
        output_dir: 输出文件夹路径
        threshold: 运动变化阈值（数值越大越敏感）
    """
    cap = cv2.VideoCapture(video_path)
    success, prev_frame = cap.read()
    frame_count = 0
    
    while success:
        # 转换为灰度图降低计算量
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        success, curr_frame = cap.read()
        if not success:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算两帧之间的差异
        diff = cv2.absdiff(curr_gray, prev_gray)
        if cv2.countNonZero(diff) > threshold:
            # 保存关键帧
            filename = f"{output_dir}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(filename, curr_frame)
            frame_count += 1
        
        prev_frame = curr_frame.copy()

    cap.release()
    print(f"提取到{frame_count}张关键帧")

# 使用示例
extract_key_frames("nike.mp4", "../train_data/datou", threshold=50)