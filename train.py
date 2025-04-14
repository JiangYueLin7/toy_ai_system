import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import json
import os

# 配置参数
IMG_SIZE = 224       # 提高分辨率提升细节识别
BATCH_SIZE = 32
EPOCHS = 30
DATA_PATH = "train_data"
CLASS_NUM = len(os.listdir(DATA_PATH))  # 自动获取类别数

# 增强数据预处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.7,1.3],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 数据流生成
train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# 使用更强大的预训练模型
base_model = EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# 优化模型结构
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(CLASS_NUM, activation='softmax')
])

# 学习率调度
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=100,
    decay_rate=0.96
)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练并保存
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=train_generator.validation_data
)

# 保存模型和类别
model.save('trained_models/toy_model.h5')
with open('trained_models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

print(f"训练完成！验证准确率：{history.history['val_accuracy'][-1]:.2%}")