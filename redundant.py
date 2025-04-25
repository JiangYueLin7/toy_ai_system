import cv2
import numpy as np
import os
import sys
import hashlib
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import shutil
import time
import random
from filelock import FileLock
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import imagehash


# 定义全局变量
FEATURE_CACHE = {}  # 缓存特征向量
LOCK_TIMEOUT = 10  # 文件锁超时时间（秒）
similar_pairs = []

def safe_imread(img_path):
    """增强型安全读取，支持中文/特殊字符路径"""
    try:
        img_path = os.path.normpath(img_path).replace('\\', '/')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        
        from PIL import Image
        img_pil = Image.open(img_path).convert('L')
        return np.array(img_pil)
    except Exception as e:
        print(f"🔴 读取失败: {img_path} - {str(e)}")
        return None

def validate_file_integrity(file_path):
    """检查文件完整性（MD5校验）"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"🔴 文件校验失败: {file_path} - {str(e)}")
        return None

def check_path_length(path, max_len=260):
    """检查路径长度（Windows限制260字符）"""
    return len(path) <= max_len

def dhash(image_path, hash_size=16):
    """改进的差值哈希算法"""
    image = safe_imread(image_path)
    if image is None:
        return None
    
    try:
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        # return sum([2 ​**​ i for i, v in enumerate(diff.flatten()) if v])
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception as e:
        print(f"🔴 哈希计算失败: {image_path} - {str(e)}")
        return None

def feature_extraction(image_path):
    """使用ResNet50提取深度特征"""
    try:
        model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        img = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return model.predict(img_array).flatten()
    except Exception as e:
        print(f"🔴 特征提取失败: {image_path} - {str(e)}")
        return None

def get_leaf_directories(root_dir):
    """获取所有最底层子目录"""
    leaf_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:
            leaf_dirs.append(dirpath)
    return leaf_dirs

def copy_similar_images(src_dir, dst_dir, threshold=0.99):
    """复制相似度高于阈值的图片到目标目录"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # 收集所有图片特征
    features = {}
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                if check_path_length(img_path) and validate_file_integrity(img_path):
                    feature = feature_extraction(img_path)
                    if feature is not None:
                        features[img_path] = feature
    
    # 计算相似度矩阵
    img_paths = list(features.keys())
    if len(img_paths) < 2:
        print("📝 未找到足够图片进行比较")
        return
    
    sim_matrix = cosine_similarity(list(features.values()))
    
    # 查找相似图片对
    
    for i in range(len(sim_matrix)):
        for j in range(i+1, len(sim_matrix)):
            if sim_matrix[i][j] > threshold:
                similar_pairs.append((img_paths[i], img_paths[j]))
    
    # 复制相似图片到目标目录
    print(f"\n发现 {len(similar_pairs)} 对相似图片，开始复制...")
    for pair in tqdm(similar_pairs, desc="复制相似图片"):
        src1, src2 = pair
        
        # 创建子目录结构
        rel_path = os.path.relpath(os.path.dirname(src1), src_dir)
        dst_subdir = os.path.join(dst_dir, rel_path)
        os.makedirs(dst_subdir, exist_ok=True)
        
        # 复制文件
        shutil.copy2(src1, dst_subdir)
        shutil.copy2(src2, dst_subdir)
    
    print(f"\n完成！共复制 {len(similar_pairs)*2} 张图片到 {dst_dir}")

def generate_report(similar_pairs, report_dir):
    """生成可视化对比报告"""
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    for idx, (img1, img2) in enumerate(similar_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Image 1: {os.path.basename(img1)}")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Image 2: {os.path.basename(img2)}")
        axes[1].axis('off')
        
        plt.savefig(os.path.join(report_dir, f"report_{idx}.png"))
        plt.close()

def main():
    if len(sys.argv) < 3:
        print("用法: python deduplicate.py <输入根目录> <输出目录> [相似度阈值]")
        print("示例: python deduplicate.py ./input ./output 0.85")
        sys.exit(1)
    
    input_root = sys.argv[1]
    output_root = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv)>=4 else 0.9
    
    # 执行相似图片检测
    copy_similar_images(input_root, output_root, threshold)
    
    # 生成对比报告
    report_dir = os.path.join(output_root, "similarity_report")
    generate_report(similar_pairs, report_dir)
    
    print("\n所有操作已完成！")

if __name__ == "__main__":
    main()