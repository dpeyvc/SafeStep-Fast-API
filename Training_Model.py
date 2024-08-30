import pandas as pd
import numpy as np
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. 데이터 읽기 및 전처리
# CSV 파일 경로 설정
data_path = './train/dataset.csv'

# 데이터 읽기
data = pd.read_csv(data_path)

# 가속도계와 자이로스코프 데이터 컬럼과 활동 레이블 선택
X = data[['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
y = data['activity'].values

# 레이블 인코딩 (정지 = 0, 걷기 = 1, 뛰기 = 2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 데이터를 훈련 및 검증 데이터로 분리 (80% 훈련, 20% 검증)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 데이터 리쉐이핑 (CNN 입력 형태로 변환)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# 2. 모델 정의 (CNN 사용)
model = models.Sequential([
    layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.Conv1D(64, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')  # 3개의 클래스 (정지, 걷기, 뛰기)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. 모델 학습 및 성능 평가
epochs = 50
batch_size = 32

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# 검증 데이터에 대한 예측
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=-1)

# 성능 평가
accuracy = np.mean(y_pred_classes == y_val)
print(f"Validation Accuracy: {accuracy:.8f}")

# 모델 저장
model.save("sensor_model.h5")
print("Keras model saved to sensor_model.h5")

# 모델 요약
model.summary()

# 4. 학습 과정 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 5. 가속도계 임계값을 적용하여 '뛰기'로 판단
def adjust_predictions_with_acceleration(X, y_pred_classes, acceleration_threshold):
    adjusted_predictions = y_pred_classes.copy()
    for i in range(len(y_pred_classes)):
        if y_pred_classes[i] == 1:  # 모델이 '걷기'로 예측한 경우
            accel_magnitude = np.sqrt(X[i, 0]**2 + X[i, 1]**2 + X[i, 2]**2)
            if accel_magnitude > acceleration_threshold:
                adjusted_predictions[i] = 2  # '뛰기'로 재조정
    return adjusted_predictions

# 가속도계 임계값 설정 (이 값은 데이터에 따라 조정 필요)
acceleration_threshold = 1.5

# 검증 데이터에 대한 조정된 예측
adjusted_y_pred_classes = adjust_predictions_with_acceleration(X_val.reshape(X_val.shape[0], X_val.shape[1]), y_pred_classes, acceleration_threshold)

# 조정된 성능 평가
adjusted_accuracy = np.mean(adjusted_y_pred_classes == y_val)
print(f"Adjusted Validation Accuracy: {adjusted_accuracy:.8f}")
