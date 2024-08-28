import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 예측에 사용할 데이터 읽기
test_data_path = 'test/X_test.txt'  # 테스트 데이터 파일 경로
test_labels_path = 'test/y_test.txt'  # 테스트 레이블 파일 경로

X_test = pd.read_csv(test_data_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(test_labels_path, delim_whitespace=True, header=None)

# 데이터 변환
X_test = np.array(X_test)

# 레이블 인코딩
le = LabelEncoder()
y_test = le.fit_transform(y_test.values.ravel())

# 데이터 리쉐이핑
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 저장된 모델 로드
model = load_model('cnn_har_model.h5')

# 모델을 사용하여 예측 수행
y_pred = np.argmax(model.predict(X_test), axis=-1)

# 상태 매핑 정의 (0 추가)
label_mapping = {
    0: "STOPPING",  # 0에 대한 예외 처리
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# 예측 결과를 상태로 변환
predicted_states = [label_mapping[pred] if pred in label_mapping else "INVALID" for pred in y_pred]

# 첫 번째 상태 출력
print(f"First predicted state: {predicted_states[0]}\n")

# 예측 결과를 10개씩 묶어서 분석
batch_size = 10
for i in range(0, len(predicted_states), batch_size):
    batch_states = predicted_states[i:i + batch_size]

    # 각 상태의 수를 계산
    state_counts = {state: batch_states.count(state) for state in set(batch_states)}

    print(f"Batch {i // batch_size + 1} Analysis:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} times")
    print("\n")

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy: {accuracy:.8f}")
print(f"Test Precision: {precision:.8f}")
print(f"Test Recall: {recall:.8f}")
print(f"Test F1-Score: {f1:.8f}")