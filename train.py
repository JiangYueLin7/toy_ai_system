import tensorflow as tf
import json
import os

# 配置参数
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
DATA_PATH = "train_data"

# 手动构建 class_indices
class_names = sorted(os.listdir(DATA_PATH))
class_indices = {name: idx for idx, name in enumerate(class_names)}

# 创建数据集（修正版）
def load_data_custom(DATA_PATH, IMG_SIZE, BATCH_SIZE, subset):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_PATH,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset=subset,  # 明确指定子集类型
        interpolation='bilinear'
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 分别加载训练集和验证集
train_dataset = load_data_custom(DATA_PATH, IMG_SIZE, BATCH_SIZE, subset='training')
val_dataset = load_data_custom(DATA_PATH, IMG_SIZE, BATCH_SIZE, subset='validation')

# 构建模型（保持原有结构）
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax', dtype='float32')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 配置回调
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

# 训练模型（修正参数传递）
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# 保存模型和类别映射
model.save('trained_models/mobile_model.keras')  # 推荐使用新格式
with open('trained_models/class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print(f"训练完成！最佳验证准确率：{max(history.history['val_accuracy']):.2%}")