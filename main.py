from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import csv
import os
import asyncio

app = FastAPI()

# 데이터 저장용 리스트
sensor_data = []
file_counter = 0

# CSV 파일을 저장할 폴더 이름
folder_name = 'csv'

# 폴더가 존재하지 않으면 생성
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 클라이언트 서버 웹소켓 연결을 저장할 변수
client_websocket = None


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    global file_counter, client_websocket
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
                    sensor_data.append(data_list)
                    print(f"Processed data: {data_list}")

                    # 클라이언트 서버로 데이터 전송
                    if client_websocket:
                        await client_websocket.send_text(data)

                    # 데이터 수집 후 일정량 이상이 되면 CSV 파일로 저장
                    if len(sensor_data) >= 5000:
                        await save_to_csv()
                        sensor_data.clear()
                else:
                    print(f"Invalid data format: {data}")

    except WebSocketDisconnect:
        await save_to_csv()
        if websocket == client_websocket:
            client_websocket = None
        else:
            print("App disconnected")
    except Exception as e:
        print(f"Error processing data: {e}")


async def save_to_csv():
    global file_counter
    filename = f'sensor_data_{file_counter}.csv'
    file_path = os.path.join(folder_name, filename)

    try:
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['device_id', 'latitude', 'longitude', 'gyro_x', 'gyro_y', 'gyro_z', 'timestamp']
            writer = csv.writer(csvfile)

            writer.writerow(fieldnames)
            writer.writerows(sensor_data)

        print(f"Data saved to {file_path}")
        file_counter += 1
    except Exception as e:
        print(f"Error saving CSV file: {e}")


@app.get("/")
async def root():
    return {"message": "WebSocket server is running"}