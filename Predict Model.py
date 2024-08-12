import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

# 예측 함수
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# 메인 실행 코드
if __name__ == "__main__":
    # 저장된 모델 로드
    model = tf.keras.models.load_model('gyroscope_model')

    scaler = StandardScaler()

    # 예측
    input_data = np.array([[0, -10, 10.0],
                           [5, 0, -5.0],
                           [-5, 5, 5.0]])

    scaled_input = scaler.fit_transform(input_data)
    print("Scaled Input Data:")
    print(scaled_input)