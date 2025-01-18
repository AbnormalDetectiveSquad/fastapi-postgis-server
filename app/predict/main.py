import logging
import warnings
import sys
sys.path.append('../../')
sys.path.append('../')
from predict.model import utility as U
from app.database import get_db
from app.models import ItsTrafficData, KmaWeatherData, LinkGridMapping, TrafficPrediction
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 설정 파일 읽기 
def read_config():
   config = U.configparser.ConfigParser()
   config.read('./model/config.ini')
   
   # 읽은 값 출력해보기
   print(f"총 배열 개수: {config['arrays']['number']}")   
   return config


# 단독실행 테스트 코드 (추후 async 및 db commit 추가)
def predict_model():
    db = next(get_db())
    # 모델 실행시점 (임시 지정)
    now = datetime(2024, 10, 1, 2, 0)
    two_hours_ago = now - timedelta(hours=2)

    # 모델 실행시점 이전 데이터 조회
    # !! 부등호 맞춰서 수정해야 함 !!
    # 1. 교통 데이터 조회 
    traffic_rows = db.query(ItsTrafficData.tm, 
                          ItsTrafficData.link_id, 
                          ItsTrafficData.speed)\
                    .filter(ItsTrafficData.tm.between(two_hours_ago, now))\
                    .all()

    df_traffic = pd.DataFrame(traffic_rows, columns=['tm', 'link_id', 'speed'])

    # 2. 기상 데이터 조회
    weather_rows = db.query(KmaWeatherData.tm, 
                          KmaWeatherData.nx, 
                          KmaWeatherData.ny, 
                          KmaWeatherData.pty, 
                          KmaWeatherData.rn1)\
                    .filter(KmaWeatherData.tm.between(two_hours_ago, now))\
                    .all()

    df_weather = pd.DataFrame(weather_rows, columns=['tm', 'nx', 'ny', 'pty', 'rn1'])

    # 3. 링크 그리드 매핑 데이터 조회
    link_mapping = db.query(LinkGridMapping.link_id, LinkGridMapping.nx, LinkGridMapping.ny).all()
    df_mapping = pd.DataFrame(link_mapping, columns=['link_id', 'nx', 'ny'])

    df_traffic_with_grid = df_traffic.merge(df_mapping, on='link_id', how='inner')

    # 4. 교통 데이터와 기상 데이터 결합
    df_combined = df_traffic_with_grid.merge(
        df_weather,
        on=['nx', 'ny', 'tm'],
        how='left'
    )

    # 5. 교통 데이터와 기상 데이터 피벗
    df_combined_pivot = df_combined.melt(
        id_vars=['link_id', 'tm'],
        value_vars=['speed', 'pty', 'rn1']
    ).pivot_table(
        index='link_id',
        columns=['tm', 'variable'],
        values='value'
    ).reset_index()

    target_time = datetime(2024, 10, 1, 1, 0)
    print(df_combined_pivot[df_combined_pivot['link_id'] == '1030022200'][(target_time, 'speed')])

    # 요일 정보 계산
    today = datetime.now(tz=ZoneInfo("Asia/Seoul"))
    weekday = today.weekday()
    if weekday == 0:
        weakday = 0
    elif weekday == 4:
        weakday = 0.5
    elif weekday in [5,6]:
        weakday = 1
    else:
        weakday = 0.1

    reader = U.Datareader()

    result = reader.process_data(A,B,C,D,E,F)

    # 예측 결과 저장 (created_at:UTC, 예측 결과:result)
    result['created_at'] = datetime.now()  # UTC 시간
    result = result.rename(columns=
        {
            'Link_ID': 'link_id',
            '5 min': 'prediction_5min',
            '10 min': 'prediction_10min',
            '15 min': 'prediction_15min'
        }
    )

    # 예측 결과 저장
    db.add(TrafficPrediction(
        tm=now,
        link_id=result['link_id'],
        prediction_5min=result['prediction_5min'],
        prediction_10min=result['prediction_10min'],
        prediction_15min=result['prediction_15min']
    ))
    db.commit()

if __name__ == "__main__":

    y=U.TestScript()
    print(y)
    print(y.shape)
    reader=U.Datareader(option='test')
    # test array 추가
    A,B,C,D,E,F=reader.testarrays.copy()
    print(f'test data:{A},{B},{C},{D},{E},{F}')
    out=reader.process_data(A,B,C,D,E,F)
    print (f'outdata:{out}')
    print (out.shape)