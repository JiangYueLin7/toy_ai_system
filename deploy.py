import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time

# 定义与训练模型一致的预处理方式
def custom_preprocess(img):
    img = tf.image.resize(img, (224, 224))  # 与训练时的IMG_SIZE保持一致
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)  # 使用对应模型的预处理
    return img

# 加载模型时保持自定义层定义（如果模型结构有变化需要检查是否需要Cast层）
class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, dtype=tf.float32)

# 其他配置保持不变
MODEL_PATH = "trained_models/mobile_model.keras"  # 注意模型格式已改为keras原生格式
LABEL_PATH = "trained_models/class_indices.json"
CROP_RECT = (400, 200, 800, 600)  # 保持与实际检测区域一致

# 加载模型和标签（保持自定义层定义）
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Cast': Cast})
with open(LABEL_PATH, 'r') as f:
    class_indices = json.load(f)
    classes = list(class_indices.keys())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    x1, y1, x2, y2 = CROP_RECT
    roi = frame[y1:y2, x1:x2]
    
    # 修改为与训练模型一致的预处理
    img = custom_preprocess(roi)
    img_array = tf.expand_dims(img, axis=0)
    
    start_time = time.time()
    predictions = model.predict(img_array)
    pred_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    fps = 1 / (time.time() - start_time)
    
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"{pred_class} ({confidence:.0%})", 
               (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Toy Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()