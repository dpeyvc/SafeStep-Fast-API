from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tensorflow as tf
import numpy as np

app = FastAPI()

# TensorFlow 모델 로드
model = tf.keras.models.load_model('cnn_har_model.keras')

# 클라이언트 서버 웹소켓 연결을 저장할 변수
client_websocket = None

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    global client_websocket
    await websocket.accept()
    print("Connection established")

    try:
        while True:
            data = await websocket.receive_text()

            if data == "0":
                print("Received '0'. This is the client server.")
                client_websocket = websocket
                await client_websocket.send_text("Ready to send sensor data")
            else:
                # 앱에서 온 데이터 처리
                data_list = data.split(',')
                if len(data_list) == 7:
                    try:
                        # 예측을 위해 데이터를 float으로 변환
                        data_float = list(map(float, data_list))
                        print(f"Processed data: {data_float}")

                        # 예측을 위한 데이터 준비 (모델이 필요로 하는 형식에 맞게 reshape)
                        data_array = np.array([data_float])  # 데이터의 배열 크기 조정 필요
                        data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)  # 모델 입력에 맞게 리쉐이핑

                        # 모델 예측
                        prediction = model.predict(data_array)
                        predicted_class = np.argmax(prediction)  # 예측된 클래스

                        print(f"Predicted class: {predicted_class}")

                        # 클라이언트 서버로 예측 결과 전송
                        if client_websocket:
                            await client_websocket.send_text(f"Prediction: {predicted_class}")
                    except Exception as e:
                        print(f"Error processing data for prediction: {e}")
                else:
                    print(f"Invalid data format: {data}")

    except WebSocketDisconnect:
        if websocket == client_websocket:
            client_websocket = None
        print("Client disconnected")
    except Exception as e:
        print(f"Error processing data: {e}")


