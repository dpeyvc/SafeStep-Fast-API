from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List
import pandas as pd
import os
import json

app = FastAPI()

class ConnectionManager:
    def __init__(self, max_entries: int):
        self.active_connections: List[WebSocket] = []
        self.gyro_data = []
        self.location_data = []
        self.max_entries = max_entries

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("클라이언트가 연결되었습니다.")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("클라이언트가 연결 해제되었습니다.")

    async def receive_and_save(self, websocket: WebSocket):
        try:
            while True:
                data = await websocket.receive_text()
                # JSON 문자열 파싱
                data_dict = json.loads(data)

                # 수신된 데이터가 딕셔너리 형태인지 확인
                if isinstance(data_dict, dict):
                    data_dict = [data_dict]  # 단일 객체를 리스트로 변환

                # 자이로스코프 데이터 저장
                gyro_data = [d['data'] for d in data_dict if d['type'] == 'gyroscope']
                if gyro_data:
                    self.gyro_data.extend(gyro_data)
                    print("수신된 자이로스코프 데이터:", gyro_data)  # 자이로스코프 데이터 출력

                # 위치 데이터 저장
                location_data = [d['data'] for d in data_dict if d['type'] == 'location']
                if location_data:
                    self.location_data.extend(location_data)
                    print("수신된 위치 데이터:", location_data)  # 위치 데이터 출력

                # 데이터의 수가 max_entries를 넘으면 파일 저장
                if len(self.gyro_data) >= self.max_entries:
                    gyro_df = pd.DataFrame(self.gyro_data[:self.max_entries])
                    gyro_df.to_csv('gyroscope_data.csv', index=False)
                    # 데이터를 파일에 저장한 후 리스트 비우기
                    self.gyro_data = self.gyro_data[self.max_entries:]

                if len(self.location_data) >= self.max_entries:
                    location_df = pd.DataFrame(self.location_data[:self.max_entries])
                    location_df.to_csv('location_data.csv', index=False)
                    # 데이터를 파일에 저장한 후 리스트 비우기
                    self.location_data = self.location_data[self.max_entries:]

                # max_entries만큼 저장했다면 while 루프 종료
                if len(self.gyro_data) == 0 and len(self.location_data) == 0:
                    break

        except WebSocketDisconnect:
            self.disconnect(websocket)

# 각 CSV 파일에 저장할 최대 엔트리 수
MAX_ENTRIES = 5000

manager = ConnectionManager(max_entries=MAX_ENTRIES)

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.receive_and_save(websocket)