import tensorflow as tf
import numpy as np
import cv2
from scipy.spatial.distance import cosine

# 저장된 MobileNetV2 모델 불러오기
model_save_path = 'mobilenetv2_model.h5'
model = tf.keras.models.load_model(model_save_path)

def preprocess_image(image_path):
    print(f"Loading image from: {image_path}")  # 이미지 경로 출력
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def get_image_embedding(image_path):
    image = preprocess_image(image_path)
    embedding = model.predict(image)
    return embedding.flatten()  # 1차원 벡터로 변환

def calculate_similarity(image_path1, image_path2):
    embedding1 = get_image_embedding(image_path1)
    embedding2 = get_image_embedding(image_path2)
    
    # 코사인 유사도 계산 (1 - 코사인 거리)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

image_path1 = 'img01.jpg'
image_path2 = 'img02.jpg'

similarity_score = calculate_similarity(image_path1, image_path2)
print(f"Similarity Score: {similarity_score}")