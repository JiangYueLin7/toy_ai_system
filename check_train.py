# check_train.py
import sys
import random
import numpy as np
from common import create_data_generator, build_base_model, create_minimal_model
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def validate_pipeline(data_path, img_size=450, batch_size=8, epochs=1):
    """端到端验证管道"""
    
    # 1. 数据验证
    try:
        data_gen = create_data_generator(
            data_path=data_path,
            target_size=img_size,
            batch_size=batch_size,
            validation_split=0.0  # 不需要验证集
        )
        
        # 验证至少存在10张图片
        all_samples = []
        for _, _, files in os.walk(data_path):
            all_samples.extend(files)
        if len(all_samples) < 10:
            raise FileNotFoundError(f"数据集需至少包含10张图片（当前{len(all_samples)}张）")
            
    except Exception as e:
        print(f"🔴 数据验证失败: {str(e)}")
        return False
    
    # 2. 模型验证
    try:
        # 创建简化版模型
        base_model = build_base_model(input_shape=(img_size, img_size, 3))
        minimal_model = create_minimal_model(base_model, num_classes=2)  # 假设2类
        
        # 编译模型（使用极简配置）
        minimal_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    except Exception as e:
        print(f"🔴 模型构建失败: {str(e)}")
        return False
    
    # 3. 训练验证
    try:
        # 随机采样10张图片
        sample_files = random.sample(all_samples, 10)
        sample_x = []
        sample_y = []
        for file in sample_files:
            img = tf.keras.preprocessing.image.load_img(
                file,
                target_size=(img_size, img_size)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            sample_x.append(img_array)
            # 假设文件名包含类别信息（需根据实际情况修改）
            label = int(file.split('/')[-2])
            sample_y.append([1 if i == label else 0 for i in range(2)])
            
        sample_x = np.array(sample_x)
        sample_y = np.array(sample_y)
        
        # 执行快速训练
        history = minimal_model.fit(
            x=sample_x,
            y=sample_y,
            epochs=epochs,
            verbose=1,
            callbacks=[
                EarlyStopping(patience=1),
                ReduceLROnPlateau(patience=1)
            ]
        )
        
        # 验证损失收敛
        if np.isnan(history.history['loss'][0]):
            raise ValueError("训练损失出现NaN")
            
    except Exception as e:
        print(f"🔴 训练验证失败: {str(e)}")
        return False
    
    print("\n✅ 验证通过！可以安全执行完整训练")
    return True

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "train_data"
    success = validate_pipeline(data_path)
    exit(0 if success else 1)