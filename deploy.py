import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time

# Define custom Cast layer
class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, dtype=tf.float32)

# 配置参数
MODEL_PATH = "trained_models/toy_model.h5"
LABEL_PATH = "trained_models/class_indices.json"
CROP_RECT = (400, 200, 800, 600)  # 根据实际拍摄框调整 (x1,y1,x2,y2)

# 加载模型和标签
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Cast': Cast})
with open(LABEL_PATH, 'r') as f:
    class_indices = json.load(f)
    classes = list(class_indices.keys())

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # [4,6](@ref)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 裁剪识别区域
    x1, y1, x2, y2 = CROP_RECT
    roi = frame[y1:y2, x1:x2]
    
    # 预处理
    img = cv2.resize(roi, (450, 450))
    img_array = tf.keras.applications.efficientnet.preprocess_input(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 推理
    start_time = time.time()
    predictions = model.predict(img_array)
    pred_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    fps = 1 / (time.time() - start_time)
    
    # 显示结果
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, f"{pred_class} ({confidence:.0%})", 
               (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Toy Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()