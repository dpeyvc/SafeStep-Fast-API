from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tensorflow as tf
import numpy as np

app = FastAPI()

# TensorFlow 모델 로드
model = tf.keras.models.load_model('cnn_har_model.h5')

# 클라이언트와 디바이스 ID를 매칭하기 위한 딕셔너리
client_connections = {}

# 데이터를 버퍼링할 딕셔너리 (디바이스 ID별로 데이터 저장)
data_buffers = {}

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    # 클라이언트와 디바이스 ID를 매칭하기 위한 변수
    device_id = None

    await websocket.accept()
    print("Connection established")

    try:
        while True:
            data = await websocket.receive_text()

            if data.startswith("ID:"):
                # 디바이스 ID 설정
                device_id = data.split(":")[1].strip()
                print(f"Device ID set to: {device_id}")

                # 디바이스 ID와 클라이언트 연결 저장
                client_connections[device_id] = websocket
                data_buffers[device_id] = []  # 새로운 디바이스에 대한 데이터 버퍼 초기화

                await websocket.send_text("Ready to send sensor data")
            else:
                if device_id is None:
                    await websocket.send_text("Device ID not set. Please send your ID first.")
                    continue

                # 앱에서 온 데이터 처리
                data_list = data.split(',')
                if len(data_list) == 7:  # 데이터가 8개의 요소를 가질 때 (시작 지점, x, y, z, ... 포함)
                    try:
                        # x, y, z 값만 추출 (5, 6, 7번째 요소)
                        data_floats = [float(data_list[i]) for i in range(5, 7)]
                        print(f"Processed data for device {device_id}: {data_floats}")

                        # 디바이스 ID에 대한 데이터 버퍼에 추가
                        data_buffers[device_id].append(data_floats)

                        # 데이터 버퍼가 10개로 가득 찼는지 확인 (10개의 x, y, z 값 세트)
                        if len(data_buffers[device_id]) >= 10:
                            # Flatten or concatenate to match the expected shape (561,)
                            flattened_data = np.array(
                                data_buffers[device_id]).flatten()  # Flatten the 10x3 data to 30 features

                            # Ensure the data has the correct shape for the model (561,)
                            if flattened_data.size < 561:
                                # Pad with zeros if not enough features
                                padded_data = np.pad(flattened_data, (0, 561 - flattened_data.size), 'constant')
                            else:
                                # Truncate if more than 561 features
                                padded_data = flattened_data[:561]

                            data_array = np.array([padded_data]).reshape(1, 561, 1)  # Reshape to (1, 561, 1)

                            # 모델 예측
                            prediction = model.predict(data_array)
                            predicted_classes = np.argmax(prediction, axis=1)  # 예측된 클래스들

                            # Convert predictions to integers and prepare the response
                            predicted_classes_int = [int(cls) for cls in predicted_classes]
                            print(f"Predicted classes for device {device_id}: {predicted_classes_int}")

                            # 클라이언트 서버로 예측 결과 전송
                            if device_id in client_connections:
                                websocket = client_connections[device_id]
                                await websocket.send_text(f"Predictions: {predicted_classes_int}")

                            # 데이터 버퍼 비우기 (모든 데이터 제거)
                            data_buffers[device_id].clear()
                        else:
                            print(f"Buffer for device {device_id} is not full. Waiting for more data.")

                    except Exception as e:
                        print(f"Error processing data for device {device_id}: {e}")
                else:
                    print(f"Invalid data format from device {device_id}: {data}")
                    await websocket.send_text("Invalid data format")
    except WebSocketDisconnect:
        if device_id in client_connections:
            del client_connections[device_id]
        if device_id in data_buffers:
            del data_buffers[device_id]
        print(f"Client with device ID {device_id} disconnected")
    except Exception as e:
        print(f"Error processing data for device {device_id}: {e}")
