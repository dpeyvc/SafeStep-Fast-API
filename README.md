# SafeStep

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-green)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)](https://www.tensorflow.org/)
[![pandas](https://img.shields.io/badge/pandas-1.5.3-blue)](https://pandas.pydata.org/)

---

## 프로젝트 개요
자이로스코프와 가속도계센서 등의 보행 데이터를 분석하여 위험 상황 감지를 목적으로 함

---


### 1. 환경 준비

- Python 3.8 이상 권장
- Git 클라이언트 설치

### 2. 레포지토리 복제

```bash
git clone https://github.com/dpeyvc/SafeStep-Fast-API.git
cd SafeStep-Fast-API
```

### 3. 가상환경 설정 및 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 데이터 및 모델 준비

1. `data/train/`, `data/test/` 폴더에 CSV 파일 배치
2. `notebooks/SafeStep_ml_module_preprocessing.ipynb` 실행하여 **모델(sensor_model.h5)** 과 **스케일러(scaler.pkl)** 생성

### 5. 서버 실행

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

- 서버 헬스체크: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔧 환경 변수

환경 변수로 설정 가능한 옵션:

| 변수 이름       | 설명                           | 기본값           |
|---------------|------------------------------|----------------|
| `BUFFER_SIZE` | 버퍼 플러시 기준 개수             | `5000`         |
| `CSV_PATH`    | 저장할 CSV 파일 경로               | `csv/predictions.csv` |
| `MODEL_PATH`  | 학습된 모델 파일 경로              | `models/sensor_model.h5` |
| `SCALER_PATH` | 저장된 Scaler 객체 경로           | `models/scaler.pkl`     |

---

## 📡 WebSocket API

### 엔드포인트

```
ws://<호스트>:8000/ws
```

### 메시지 포맷

1. **모니터링 시작**

   ```json
   {
     "type": "register",
     "role": "monitor"
   }
   ```
   - 서버 응답:
     ```json
     {
       "type": "info",
       "message": "Ready to receive sensor data"
     }
     ```

2. **디바이스 등록**

   ```json
   {
     "type": "register",
     "role": "device",
     "id": "sensor01"
   }
   ```
   - 서버 응답:
     ```json
     {
       "type": "info",
       "message": "Device sensor01 registered"
     }
     ```

3. **센서 데이터 전송**

   ```json
   {
     "type": "data",
     "values": [0.12, -0.03, 0.45, 0.33, 0.18, -0.22, 0.05]
   }
   ```
   - 서버 응답:
     ```json
     {
       "type": "prediction",
       "value": 2
     }
     ```

4. **오류 처리**

   ```json
   {
     "type": "error",
     "message": "Invalid data format"
   }
   ```

---

## 🛠️ 문제 해결 및 FAQ

- **Q: 모델 파일이 없어요.**
  A: `notebooks/SafeStep_ml_module_preprocessing.ipynb`를 실행하여 `models/` 폴더에 모델 및 스케일러를 생성하세요.

- **Q: WebSocket 연결이 거부됩니다.**
  A: 서버가 실행 중인지 확인(`uvicorn` 로그) 후, URL(`ws://localhost:8000/ws`)이 정확한지 검토하세요.

- **Q: CSV 출력이 안 돼요.**
  A: `BUFFER_SIZE` 환경 변수를 줄이거나, 서버 종료 시 `Ctrl+C`로 정상 종료하여 잔여 데이터가 플러시되도록 하세요.

---
