from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import csv
import os

app = FastAPI()

# 데이터 저장용 리스트
sensor_data = []
file_counter = 0

# CSV 파일을 저장할 폴더 이름
folder_name = 'csv'

# 폴더가 존재하지 않으면 생성
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    global file_counter
    await websocket.accept()
    print("Client connected")  # 클라이언트 연결 로그
    try:
        while True:
            try:
                data = await websocket.receive_text()
                print(f"Received raw data: {data}")  # 받은 원본 데이터 로그

                # 데이터를 쉼표로 분리하여 리스트로 변환
                data_list = data.split(',')
                if len(data_list) == 5:  # x, y, z, latitude, longitude
                    sensor_data.append(data_list)
                    print(f"Processed data: {data_list}")  # 처리된 데이터 로그

                    # 데이터 수집 후 일정량 이상이 되면 CSV 파일로 저장
                    if len(sensor_data) >= 5000:
                        await save_to_csv()
                        sensor_data.clear()
                else:
                    print(f"Invalid data format: {data}")  # 잘못된 데이터 형식 로그

                await websocket.send_text(f"Sensor data received: {data}")
            except Exception as e:
                print(f"Error processing data: {e}")  # 데이터 처리 중 오류 로그
    except WebSocketDisconnect:
        print("Client disconnected")  # 클라이언트 연결 해제 로그


async def save_to_csv():
    global file_counter
    # CSV 파일 저장 경로 지정
    filename = f'sensor_data_{file_counter}.csv'
    file_path = os.path.join(folder_name, filename)

    try:
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['gyro_x', 'gyro_y', 'gyro_z', 'latitude', 'longitude']
            writer = csv.writer(csvfile)

            writer.writerow(fieldnames)
            writer.writerows(sensor_data)

        print(f"Data saved to {file_path}")
        file_counter += 1
    except Exception as e:
        print(f"Error saving CSV file: {e}")  # CSV 파일 저장 중 오류 로그


@app.get("/")
async def root():
    return {"message": "WebSocket server is running"}
