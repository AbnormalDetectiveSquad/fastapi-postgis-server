import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db
from app.models import KmaWeatherData, ItsTrafficData
from datetime import datetime, timedelta


def check_recent_data():
    db = next(get_db())

    # 최근 1시간 데이터 확인
    one_hour_ago = datetime.now() - timedelta(hours=1)

    print("=== 최근 날씨 데이터 ===")
    weather_data = db.query(KmaWeatherData) \
        .filter(KmaWeatherData.tm >= one_hour_ago) \
        .all()
    for data in weather_data:
        print(f"시간: {data.tm}, 좌표: ({data.nx}, {data.ny}), "
              f"강수형태: {data.pty}, 강수량: {data.rn1}")

    print("\n=== 최근 교통 데이터 ===")
    st = time.time()
    traffic_data = db.query(ItsTrafficData) \
        .filter(ItsTrafficData.tm >= one_hour_ago) \
        .limit(5) \
        .all()
    for data in traffic_data:
        print(f"시간: {data.tm}, 링크ID: {data.link_id}, "
              f"속도: {data.speed}, 소요시간: {data.travel_time}")
    print(time.time()-st)

if __name__ == "__main__":
    check_recent_data()