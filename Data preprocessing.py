import pandas as pd
import numpy as np


def determine_label(std_dev):
    # 표준 편차 기준으로 상태를 결정
    if std_dev < 0.5:
        return 0  # 정지 상태
    elif 0.5 <= std_dev < 2.0:
        return 1  # 걷는 상태
    else:
        return 2  # 뛰는 상태


def preprocess_and_label_data(file_paths, sampling_rate=50, interval=25):
    labeled_data = []

    for file_path in file_paths:
        # CSV 파일 읽기
        data = pd.read_csv(file_path)

        # 각 축의 표준 편차 계산
        std_dev_x = data['x'].std()
        std_dev_y = data['y'].std()
        std_dev_z = data['z'].std()

        # 전체 표준 편차의 평균을 사용하여 라벨 결정
        mean_std_dev = np.mean([std_dev_x, std_dev_y, std_dev_z])
        label = determine_label(mean_std_dev)

        # 주어진 샘플링 간격(interval)으로 데이터를 리샘플링
        resampled_data = data.groupby(np.arange(len(data)) // interval).mean()

        # 상태 레이블 추가
        resampled_data['state'] = label

        labeled_data.append(resampled_data)

    # 모든 데이터를 하나로 합치기
    all_data = pd.concat(labeled_data, ignore_index=True)
    return all_data


def main():
    # 파일 경로 리스트
    file_paths = ['.csv']

    # 데이터를 읽어와서 라벨링 처리
    all_data = preprocess_and_label_data(file_paths, sampling_rate=50, interval=25)

    # 학습용 데이터로 분리
    x_data = all_data[['x', 'y', 'z']].values
    y_data = all_data['state'].values

    # 데이터 저장
    np.save('npy/x_data.npy', x_data)
    np.save('npy/y_data.npy', y_data)

    print("Data preprocessing complete and saved to x_data.npy and y_data.npy")


if __name__ == "__main__":
    main()