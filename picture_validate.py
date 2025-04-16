import os
import json
import shutil

def validate_dataset(image_dir, label_path):
    valid_images = set()
    with open(label_path, 'r') as f:
        labels = json.load(f)
    
    # 检查图片是否存在
    for img_info in labels:
        img_path = os.path.join(image_dir, img_info['filename'])
        if not os.path.exists(img_path):
            print(f"Invalid image: {img_path}")
            labels.remove(img_info)
            continue
        valid_images.add(img_info['filename'])
    
    # 删除无效标签
    new_labels = [img for img in labels if img['filename'] in valid_images]
    with open(label_path, 'w') as f:
        json.dump(new_labels, f)
    
    if not valid_images:
        print("No valid images found. Deleting label file.")
        os.remove(label_path)
        return False
    return True

# 使用示例
if not validate_dataset('images/', 'labels.json'):
    exit("Dataset invalid, training aborted.")