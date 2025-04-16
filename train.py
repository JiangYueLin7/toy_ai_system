import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
from common import create_data_generator, build_base_model, create_minimal_model


# 开启混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 配置参数
IMG_SIZE = 450       # 进一步提高分辨率
BATCH_SIZE = 8       # 由于图像尺寸增大，减小批次大小
EPOCHS = 100
DATA_PATH = "train_data"
CLASS_NUM = len(os.listdir(DATA_PATH))  # 自动获取类别数

# 增强数据预处理
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=60,
#     width_shift_range=0.4,
#     height_shift_range=0.4,
#     brightness_range=[0.5, 1.5],
#     shear_range=0.4,
#     zoom_range=0.4,
#     horizontal_flip=True,
#     vertical_flip=True,
#     validation_split=0.2
# )

train_datagen = create_data_generator(DATA_PATH, batch_size=BATCH_SIZE)

# 生成训练集和验证集
train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 使用更强大的预训练模型
# base_model = EfficientNetB5(
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     include_top=False,
#     weights='imagenet'
# )

base_model = build_base_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

# 先冻结所有层进行训练
base_model.trainable = False

# 优化模型结构
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(CLASS_NUM, activation='softmax', dtype='float32')  # 输出层使用 float32
])

# 学习率调度
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005,
    decay_steps=EPOCHS * len(train_generator)
)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 早停策略和学习率衰减
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.000001
)

# 第一阶段训练
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# 解冻部分层进行微调
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 创建新的学习率调度，初始学习率为原来的 1/10
fine_tune_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0005 / 10,  # 初始学习率降为原来的 1/10
    decay_steps=EPOCHS * len(train_generator)
)

# 重新编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 第二阶段训练
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# 保存模型和类别
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')
model.save('trained_models/toy_model.h5')
with open('trained_models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

print(f"训练完成！验证准确率：{max(history_fine.history['val_accuracy']):.2%}")