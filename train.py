import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
from common import create_data_generator, build_base_model, create_minimal_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# 开启混合精度训练（可选，根据显卡显存调整）
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 配置参数
IMG_SIZE = 450         # 输入图像分辨率
BATCH_SIZE = 8         # 批次大小（建议 >=8）
EPOCHS = 100           # 总训练轮次
DATA_PATH = "train_data"  # 数据集根目录
CLASS_NUM = len([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])  # 动态计算类别数

# 创建数据生成器（仅配置参数，不立即加载数据）
train_datagen = create_data_generator(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# 生成训练集和验证集的迭代器
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

# 构建基础模型
base_model = build_base_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))

# 冻结基础模型所有层
base_model.trainable = False

# 构建完整模型
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(2048, activation='relu'),
    Dropout(0.3),
    Dense(CLASS_NUM, activation='softmax', dtype='float32')  # 输出层必须与类别数一致
])

# 编译模型（使用固定学习率）
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 固定学习率
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 定义回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.000001,
    verbose=1  # 打印学习率调整信息
)

# 第一阶段训练（冻结基础模型）
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# 解冻部分基础模型层（微调）
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.8)  # 解冻最后 20% 层
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 重新编译模型（学习率自动由 ReduceLROnPlateau 管理）
model.compile(
    optimizer=tf.keras.optimizers.Adam(),  # 不再需要指定学习率
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 第二阶段训练（微调）
history_fine = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=history.epoch[-1],  # 从上一阶段最后 epoch 继续
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# 保存模型和类别映射
os.makedirs('trained_models', exist_ok=True)
model.save('trained_models/toy_model.h5')
with open('trained_models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

print(f"训练完成！最佳验证准确率：{max(history_fine.history['val_accuracy']):.2%}")