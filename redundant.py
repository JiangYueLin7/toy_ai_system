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

def safe_imread(img_path):
    """增强型安全读取，支持中文/特殊字符路径"""
    try:
        # 规范化路径格式
        img_path = os.path.normpath(img_path).replace('\\', '/')
        
        # 尝试OpenCV读取
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        
        # OpenCV失败时使用Pillow读取
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
    if len(path) > max_len:
        print(f"⚠️ 路径过长: {path} ({len(path)} > {max_len})")
        return False
    return True

def dhash(image_path, hash_size=16):
    """改进的差值哈希算法"""
    image = safe_imread(image_path)
    if image is None:
        return None
    
    try:
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        # return sum([2 ​**​ i for i, v in enumerate(diff.flatten()) if v])
        return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])
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

def deduplicate_images(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有最底层子目录
    leaf_dirs = get_leaf_directories(root_dir)
    if not leaf_dirs:
        print("⚠️ 未找到有效子目录")
        return
    
    # 收集所有有效图片
    all_images = []
    for leaf_dir in leaf_dirs:
        for entry in os.scandir(leaf_dir):
            if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = entry.path
                if check_path_length(img_path) and validate_file_integrity(img_path):
                    all_images.append(img_path)
    
    if not all_images:
        print("⚠️ 未找到有效图片")
        return
    
    # 哈希去重（全局）
    print("1/3 哈希去重中...")
    hash_dict = {}
    for img_path in tqdm(all_images, desc="计算哈希值"):
        img_hash = dhash(img_path)
        if img_hash is not None:
            hash_dict[img_hash] = img_path
    
    # 特征去重（按子目录分组）
    print("\n2/3 特征去重中...")
    group_features = {}
    for img_path in tqdm(hash_dict.values(), desc="提取特征"):
        # 确定所属子目录
        for leaf_dir in leaf_dirs:
            if img_path.startswith(leaf_dir):
                group_id = leaf_dir
                break
                
        feature = feature_extraction(img_path)
        if feature is None:
            continue
            
        if group_id not in group_features:
            group_features[group_id] = {
                'features': [],
                'paths': []
            }
            
        group_features[group_id]['features'].append(feature)
        group_features[group_id]['paths'].append(img_path)
    
    # 保存去重结果
    print("\n3/3 保存结果中...")
    for group_id, group_data in group_features.items():
        features = np.array(group_data['features'])
        paths = group_data['paths']
        
        # 计算相似度矩阵
        sim_matrix = cosine_similarity(features)
        np.fill_diagonal(sim_matrix, 0)  # 忽略自身
        
        # 保留相似度最低的图片
        keep_indices = []
        while len(sim_matrix) > 0:
            min_sim = np.min(sim_matrix)
            if min_sim >= 0.9:
                break  # 停止条件
            
            min_index = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
            keep_indices.append(min_index[0])
            
            # 移除已选中图片
            sim_matrix = np.delete(sim_matrix, min_index[0], axis=0)
            sim_matrix = np.delete(sim_matrix, min_index[0], axis=1)
        
        # 保存保留的图片
        for idx in keep_indices:
            src = paths[idx]
            dst = os.path.join(output_dir, os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            print(f"✅ 保留: {src} → {dst}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python deduplicate.py <输入根目录> <输出目录>")
        sys.exit(1)
    
    input_root = sys.argv[1]
    output_root = sys.argv[2]
    
    deduplicate_images(input_root, output_root)