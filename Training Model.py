import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# 모델 학습 함수
def train_model(x_train, y_train, epochs=200):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),  # 50% Dropout 추가
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # 50% Dropout 추가
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # 학습률을 낮춘 Adam 옵티마이저
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    # 모델 학습
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    # 학습된 모델 저장
    model.save('gyroscope_model', save_format='tf')
    print("Model training complete and saved to gyroscope_model")

    # 최종 학습 및 검증 정확도 출력
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

# 메인 실행 코드
if __name__ == "__main__":
    # 저장된 학습 데이터를 불러오기
    x_train = np.load('./npy/x_data.npy')
    y_train = np.load('./npy/y_data.npy')

    # 모델 학습
    train_model(x_train, y_train, epochs=200)