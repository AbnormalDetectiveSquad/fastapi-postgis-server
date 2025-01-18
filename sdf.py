import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import glob

# DB 연결
engine = create_engine('postgresql://postgres:abcd@localhost:5432/test_model')


def process_file(file_path):
    # CSV 읽기
    df = pd.read_csv(file_path,names=['tm1', 'tm2', 'link_id', 'road_authority', 'speed', 'travel_time'])
    # 첫 두 컬럼으로 timestamp 만들기
    # 20240101 + 0000 -> 2024-01-01 00:00:00
    df['tm'] = pd.to_datetime(df.iloc[:, 0].astype(str) + df.iloc[:, 1].astype(str).str.zfill(4),
                              format='%Y%m%d%H%M')

    # 필요한 컬럼만 선택
    df = df[['tm', 'link_id', 'road_authority', 'speed', 'travel_time']]

    print(f"Processing {file_path}: {len(df)} rows")

    # DB에 입력
    try:
        df.to_sql('its_traffic_data',
                  engine,
                  if_exists='append',
                  index=False,
                  method='multi',
                  chunksize=10000)
        print(f"Successfully inserted {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


# CSV 파일들 목록 가져오기
csv_files = glob.glob(r'/home/ssy/extract_its_data/*_5Min.csv')

# 각 파일 처리
q=0
for file in sorted(csv_files):
    process_file(file)
    q+=1
    if q==3:
        break