# model_analysis.py

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 모델 경로를 전역 변수로 설정
MODEL_PATH = 'cnn_har_model.h5'  # 실제 모델 파일 경로로 변경하세요


def preprocess_data(X, y):
    """
    데이터를 전처리하는 함수
    """
    le = LabelEncoder()
    y = le.fit_transform(y.ravel())
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def load_model_and_predict(input_data):
    """
    모델을 로드하고 예측을 수행하는 함수

    Parameters:
    input_data (np.ndarray): 예측에 사용할 입력 데이터

    Returns:
    np.ndarray: 예측된 레이블
    """
    # 전역 변수로 설정된 모델 경로 사용
    model = load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(input_data), axis=-1)
    return y_pred


def analyze_predictions(y_pred):
    """
    예측 결과를 분석하는 함수
    """
    label_mapping = {
        0: "UNKNOWN",
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }
    predicted_states = [label_mapping[pred] if pred in label_mapping else "INVALID" for pred in y_pred]
    return predicted_states


def print_analysis(predicted_states):
    """
    분석 결과를 출력하는 함수
    """
    print(f"First predicted state: {predicted_states[0]}\n")
    batch_size = 10
    for i in range(0, len(predicted_states), batch_size):
        batch_states = predicted_states[i:i + batch_size]
        state_counts = {state: batch_states.count(state) for state in set(batch_states)}
        print(f"Batch {i // batch_size + 1} Analysis:")
        for state, count in state_counts.items():
            print(f"  {state}: {count} times")
        print("\n")


def evaluate_performance(y_true, y_pred):
    """
    모델의 성능을 평가하는 함수
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Test Accuracy: {accuracy:.8f}")
    print(f"Test Precision: {precision:.8f}")
    print(f"Test Recall: {recall:.8f}")
    print(f"Test F1-Score: {f1:.8f}")
