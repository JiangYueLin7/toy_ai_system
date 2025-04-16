# common.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential

def create_data_generator(
    data_path,
    target_size=(450, 450),
    batch_size=8,
    class_mode='categorical',
    validation_split=0.2
):
    """创建标准化数据生成器"""
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=60,
        width_shift_range=0.4,
        height_shift_range=0.4,
        brightness_range=[0.5, 1.5],
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=validation_split
    ).flow_from_directory(
        directory=data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training'  # 默认使用训练集
    )

def build_base_model(input_shape=(450, 450, 3)):
    """构建基础EfficientNetB5模型"""
    return EfficientNetB5(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

def create_minimal_model(base_model, num_classes):
    """创建简化版模型用于验证"""
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model