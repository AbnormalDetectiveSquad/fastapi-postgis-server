import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import glob
from sqlalchemy.types import Integer, String
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

def process_file_weather():
    # CSV 읽기
    path='/home/ssy/git/traffic_prediction_model/Weather/2024_pty_rn1.csv'
    df = pd.read_csv(path,names=['nx', 'ny', 'tm', 'pty', 'rn1'])
    # 첫 두 컬럼으로 timestamp 만들기
    # 20240101 + 0000 -> 2024-01-01 00:00:00
    df['tm'] = pd.to_datetime(df.iloc[:, 2], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.iloc[1:,0]=df.iloc[1:,0].astype(float).astype(int)
    df.iloc[1:,1]=df.iloc[1:,1].astype(float).astype(int)
    df.iloc[1:,3]=df.iloc[1:,3].astype(float).astype(int)
    df.iloc[1:,4]=df.iloc[1:,4].astype(float)
    # 필요한 컬럼만 선택
    df = df[['nx', 'ny', 'tm', 'pty', 'rn1']]
    df['nx'] = pd.to_numeric(df['nx'], errors='coerce').astype('Int64')
    df['ny'] = pd.to_numeric(df['ny'], errors='coerce').astype('Int64')
    df['pty'] = pd.to_numeric(df['pty'], errors='coerce').astype('Int64')
    df['rn1'] = pd.to_numeric(df['rn1'], errors='coerce').astype('float64')
    df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna()
    print(f"Processing {path}: rows")
    print(f"df.dtypes: {df.dtypes}")
    # DB에 입력
    try:
        df.to_sql('kma_weather_data',
                  engine,
                  if_exists='append',
                  index=False,
                  method='multi',
                  chunksize=10000)
        print(f"Successfully inserted {path}")
    except Exception as e:
        print(f"Error processing {path}")



def process_file_linksort_order():
    # CSV 읽기
    path='filtered_nodes_filtered_links_table.csv'
    df = pd.read_csv(path,names=['matrix_index','link_id'])
    df.iloc[1:,1]=df.iloc[1:,1].astype(float).astype(int)
    df = df[['matrix_index','link_id']]
    df['link_id'] = pd.to_numeric(df['link_id'], errors='coerce').astype('Int64')
    df = df.dropna()
    print(f"Processing {path}: rows")
    print(f"df.dtypes: {df.dtypes}")
    # DB에 입력
    try:
        df.to_sql('link_id_sort_order',
                  engine,
                  if_exists='append',
                  index=False,
                  method='multi',
                  chunksize=10000,
                  dtype={
                    'matrix_index': Integer(),
                    'link_id': String(10)}  
        )
        print(f"Successfully inserted {path}")
    except Exception as e:
        print(f"Error processing {path}")




#process_file_weather()

process_file_linksort_order()

#CREATE TABLE link_id_sort_order(
#    matrix_index INTEGER,           
#    link_id CHARACTER VARYING(10),               
#    PRIMARY KEY (matrix_index, link_id) 
#);
## CSV 파일들 목록 가져오기
#csv_files = glob.glob(r'/home/ssy/extract_its_data/*_5Min.csv')
#
## 각 파일 처리
#q=0
#for file in sorted(csv_files):
#    process_file(file)
#    q+=1
#    if q==3:
#        break