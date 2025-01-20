import logging
import warnings
from sqlalchemy.orm import Session
from predict.model import utility as U
from models import ItsTrafficData, KmaWeatherData, LinkGridMapping, TrafficPrediction
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import holidays
import pandas as pd

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 설정 파일 읽기 
def read_config():
   config = U.configparser.ConfigParser()
   config.read('config.ini')
   
   # 읽은 값 출력해보기
   print(f"총 배열 개수: {config['arrays']['number']}")   
   return config


def predict_traffic(db: Session):
    try:
        now = datetime.now(tz=ZoneInfo("Asia/Seoul"))
        now = now.replace(
            minute=(now.minute // 5) * 5,  # 5분 단위로 절삭
            second=0,
            microsecond=0
        )

        time_intervals = [now - timedelta(minutes=5 * i) for i in range(24)]
        # 시간 데이터를 판다스 데이터프레임으로 변환
        time_series = pd.DataFrame({'Time': sorted(time_intervals)})
        # 데이터프레임 시간순 정렬 (이미 정렬된 상태)
        time_series_sorted = time_series.sort_values(by='Time').reset_index(drop=True)

        two_hours_ago = now - timedelta(hours=2)
        logging.info(f"Starting traffic prediction for time window: {two_hours_ago} to {now}")

        # 모델 실행시점 이전 데이터 조회
        # 1. 교통 데이터 조회
        traffic_rows = db.query(ItsTrafficData.tm,
                            ItsTrafficData.link_id,
                            ItsTrafficData.speed)\
                        .filter(ItsTrafficData.tm.between(two_hours_ago, now))\
                        .all()

        df_traffic = pd.DataFrame(traffic_rows, columns=['tm', 'link_id', 'speed'])
        logging.info(f"Retrieved {len(df_traffic)} traffic records")
        if len(df_traffic) == 0:
            logging.warning("No traffic data found for the specified time period")

        # 2. 기상 데이터 조회 (기상 데이터는 조회 시간을 한시간 더 이전으로 설정)
        three_hours_ago = now - timedelta(hours=3)
        logging.info(f"Fetching weather data from {three_hours_ago} to {now}")
        weather_rows = db.query(KmaWeatherData.tm,
                            KmaWeatherData.nx,
                            KmaWeatherData.ny,
                            KmaWeatherData.pty,
                            KmaWeatherData.rn1)\
                        .filter(KmaWeatherData.tm.between(three_hours_ago, now))\
                        .all()

        df_weather = pd.DataFrame(weather_rows, columns=['tm', 'nx', 'ny', 'pty', 'rn1'])
        df_weather['tm'] = pd.to_datetime(df_weather['tm'])
        logging.info(f"Retrieved {len(df_weather)} weather records")

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
        link_mapping = db.query(LinkGridMapping.link_id,
                                LinkGridMapping.nx,
                                LinkGridMapping.ny).all()
        df_mapping = pd.DataFrame(link_mapping, columns=['link_id', 'nx', 'ny'])

        df_traffic_with_grid = df_traffic.merge(df_mapping, on='link_id', how='inner')

        # 4. 교통 데이터와 기상 데이터 결합
        ## 기상데이터 5분 간격으로 리샘플 후 forward fill
        if len(df_weather) == 0:
            logging.warning("기상 데이터가 없습니다 - 기본값(0) 사용")
            # traffic 데이터의 모든 nx, ny 조합에 대해 0으로 채움
            df_weather_resampled = df_traffic_with_grid[['tm', 'nx', 'ny']].drop_duplicates()
            df_weather_resampled['pty'] = 0
            df_weather_resampled['rn1'] = 0
        else:
            df_weather_resampled = df_weather.set_index('tm')\
            .groupby(['nx', 'ny'])\
            .resample('5T')\
            .ffill()\
            .fillna({
                'pty': 0,  # (결측 시 처리) 강수형태 없음(0)
                'rn1': 0   # (결측 시 처리) 강수량 0
            })\
            .reset_index()
            logging.info(f"Resampled weather data: {len(df_weather_resampled)} records")

        df_combined = df_traffic_with_grid.merge(
            df_weather_resampled,
            on=['nx', 'ny', 'tm'],
            how='left'
        )

        # 결측치 최종 확인
        missing_pty = df_combined['pty'].isna().sum()
        missing_rn1 = df_combined['rn1'].isna().sum()
        if missing_pty > 0 or missing_rn1 > 0:
            logging.warning(f"Missing values found - PTY: {missing_pty}, RN1: {missing_rn1}")
        df_combined['pty'] = df_combined['pty'].fillna(0)
        df_combined['rn1'] = df_combined['rn1'].fillna(0)

        # 요일 정보 계산
        today = now
        # 공휴일 확인
        kr_holidays = holidays.KR()
        is_holiday = today.strftime('%Y-%m-%d') in kr_holidays

        weekday = today.weekday()
        if weekday == 0:
            weakday = 0
        elif weekday == 4:
            weakday = 0.5
        elif (weekday in [5,6]) or is_holiday:
            weakday = 1
        else:
            weakday = 0.1
        logging.info(f"Day type calculation - Weekday: {weekday}, Holiday: {is_holiday}, Weight: {weakday}")

        # 예측 수행
        logging.info("Starting prediction process")
        reader = U.Datareader()
        result = reader.process_data(df_combined, weakday)
        logging.info(f"Prediction completed for {len(result)} records")

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

        # 예측 결과를 dictionary 리스트로 변환
        predictions_to_insert = result.assign(
            tm=now
        )[['tm', 'link_id', 'prediction_5min', 'prediction_10min', 'prediction_15min', 'created_at']].to_dict('records')

        try:
            # bulk insert 실행
            db.bulk_insert_mappings(TrafficPrediction, predictions_to_insert)
            db.commit()
            logging.info(f"Successfully committed prediction result at {now}")
        except Exception as e:
            logging.error(f"Bulk insert failed: {str(e)}")
            raise

    except Exception as e:
        logging.error(f"Error in predict_traffic: {str(e)}", exc_info=True)
        db.rollback()
        raise