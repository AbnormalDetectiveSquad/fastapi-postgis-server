import logging
import warnings
import sys
import os
# 제거할 경로
path_to_remove = '/home/ssy/git/fastapi-postgis-server/app/predict'

# sys.path에서 제거
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
    print(f"Removed {path_to_remove} from sys.path")
else:
    print(f"{path_to_remove} not found in sys.path")
# app 디렉토리를 sys.path에 추가
app_path = '/home/ssy/git/fastapi-postgis-server/app'
if app_path not in sys.path:
    sys.path.append(app_path)
    print(f"Added {app_path} to sys.path")
#sys.path.append('../../')
#sys.path.append('../')
#sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print("sys.path:", sys.path)
print("Current working directory:", os.getcwd())
from predict.model import utility as U
from database import get_db
from models import ItsTrafficData, KmaWeatherData, LinkGridMapping, TrafficPrediction
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)



# 단독실행 테스트 코드 (추후 async 및 db commit 추가)
def predict_model(mainrun=False):
    db = next(get_db())
    # 모델 실행시점 (임시 지정)
    now = datetime(2023, 9, 1, 10, 0)
    nowp5 = now+timedelta(minutes=5)
    time_intervals = [now - timedelta(minutes=5 * i) for i in range(24)]
    # 시간 데이터를 판다스 데이터프레임으로 변환
    time_series = pd.DataFrame({'Time': sorted(time_intervals)})
    # 데이터프레임 시간순 정렬 (이미 정렬된 상태)
    time_series_sorted = time_series.sort_values(by='Time').reset_index(drop=True)
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

    # 2. 기상 데이터 조회 (기상데이터는 조회 시간을 한시간 더 이전으로 설정)
    three_hours_ago = now - timedelta(hours=3)
    weather_rows = db.query(KmaWeatherData.tm, 
                            KmaWeatherData.nx, 
                            KmaWeatherData.ny, 
                            KmaWeatherData.pty, 
                            KmaWeatherData.rn1)\
                    .filter(KmaWeatherData.tm.between(three_hours_ago, now))\
                    .all()

    df_weather = pd.DataFrame(weather_rows, columns=['tm', 'nx', 'ny', 'pty', 'rn1'])
    # tm이 datetime 컬럼이라고 가정
    df_weather['tm'] = pd.to_datetime(df_weather['tm'])

    def resample_each_group(sub_df):
        # 그룹 내부에서 tm을 인덱스로 잡고 리샘플링
        return (sub_df
                .set_index('tm')
                .resample('5T')
                .ffill()
            )

    df_weather_resampled = (
        df_weather
        .groupby(['nx', 'ny'], group_keys=False)
        .apply(resample_each_group)
        .reset_index()  # 인덱스를 풀어 컬럼화
    )

    # 3. 링크 그리드 매핑 데이터 조회
    link_mapping = db.query(LinkGridMapping.link_id, LinkGridMapping.nx, LinkGridMapping.ny).all()
    df_mapping = pd.DataFrame(link_mapping, columns=['link_id', 'nx', 'ny'])

    df_traffic_with_grid = df_traffic.merge(df_mapping, on='link_id', how='inner')

    # 4. 교통 데이터와 기상 데이터 결합
    df_combined = df_traffic_with_grid.merge(
        df_weather_resampled,
        on=['nx', 'ny', 'tm'],
        how='left'
    )
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
    
    print("weekday:", weekday)
 
    if not mainrun:

        reader = U.Datareader()

        result = reader.process_data(df_combined, weakday,time_series_sorted)

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

        return

    return df_combined, weakday,time_series_sorted
#
if __name__ == "__main__":
    
    Data,Holiday,Timestamp=predict_model(True)
    
    reader=U.Datareader() 
    
    output=reader.process_data(Data,Holiday,Timestamp)
    
    Result=U.calculation(output)
    
    print(Result)