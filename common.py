# 文档1改造
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large  # 改用MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential

def create_data_generator(
    data_path,
    target_size=(224, 224),  # 降低输入尺寸到224x224
    batch_size=16,  # 增加batch_size加速训练
    class_mode='categorical',
    validation_split=0.2
):
    """优化数据生成器配置"""
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # 减少数据增强强度
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],  # 缩小亮度调整范围
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,  # 移除垂直翻转（某些类别可能不适用）
        validation_split=validation_split
    )

def build_mobilenetv3_base(input_shape=(224, 224, 3)):
    """构建预训练MobileNetV3Large模型"""
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'  # 使用ImageNet预训练权重
    )
    base_model.trainable = False  # 冻结全部基础层
    return base_model

def create_optimized_model(base_model, num_classes):
    """创建轻量化模型结构"""
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),  # 增强正则化
        Dense(512, activation='relu'),  # 中间层调整
        Dropout(0.2),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # 调整学习率
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model