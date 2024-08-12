from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import csv
import asyncio
import os

app = FastAPI()

# 데이터 저장용 리스트
gyroscope_data = []
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
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            if data_json['type'] == 'gyroscope':
                gyroscope_data.append(data_json['data'])
                print(data_json['data'])

                # 데이터 수집 후 일정량 이상이 되면 CSV 파일로 저장
                if len(gyroscope_data) >= 5000:
                    await save_to_csv()
                    gyroscope_data.clear()

            await websocket.send_text(f"Gyroscope data received: {data_json['data']}")
    except WebSocketDisconnect:
        print("Client disconnected")


async def save_to_csv():
    global file_counter
    # CSV 파일 저장 경로 지정
    filename = f'gyroscope_data_{file_counter}.csv'
    file_path = os.path.join(folder_name, filename)

    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in gyroscope_data:
            writer.writerow(data)

    print(f"Data saved to {file_path}")
    file_counter += 1
