# 데이터 처리와 분석을 위한 라이브러리
import keras.models
import pandas as pd
import numpy as np

# 머신러닝을 위한 라이브러리
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 딥러닝 모델을 구축하고 학습시키기 위한 TensorFlow 및 Keras 라이브러리
from tensorflow.keras import models, layers
import tensorflow as tf

# 데이터 시각화를 위한 라이브러리
import matplotlib.pyplot as plt


# 1. 데이터 읽기 및 전처리

# 경로 설정
train_data_path = 'train/X_train.txt'
train_labels_path = 'train/y_train.txt'
test_data_path = 'test/X_test.txt'
test_labels_path = 'test/y_test.txt'

# 데이터와 레이블 읽기
X_train = pd.read_csv(train_data_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(train_labels_path, delim_whitespace=True, header=None)
X_test = pd.read_csv(test_data_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(test_labels_path, delim_whitespace=True, header=None)

# 데이터 형태 변환
X_train = np.array(X_train)
X_test = np.array(X_test)

# 레이블 인코딩
le = LabelEncoder()
y_train = le.fit_transform(y_train.values.ravel())
y_test = le.transform(y_test.values.ravel())

# 데이터 리쉐이핑 (예: CNN 입력 형태로 변환)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 2. 모델 정의 (CNN 사용 예제)

model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.5),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')  # 6개의 클래스에 대해 소프트맥스 활성화
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. 모델 학습 및 성능 평가

# 에포크 수 조절하여 모델 학습
epochs = 50  # 에포크 수 설정
batch_size = 16

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test))

# 테스트 데이터에 대한 예측
y_pred = np.argmax(model.predict(X_test), axis=-1)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy: {accuracy:.8f}")
print(f"Test Precision: {precision:.8f}")
print(f"Test Recall: {recall:.8f}")
print(f"Test F1-Score: {f1:.8f}")


model.save("cnn_har_model.h5")
model = keras.models.load_model('cnn_har_model.h5')
model.summary()
print(f"Keras model saved to cnn_har_model")


# 6. 학습 과정 시각화

# 손실 그래프
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
