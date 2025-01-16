import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import get_db
from app.fetch import fetch_data_weather, fetch_data_traffic, get_base_datetime

async def test_weather(tm=None):
    print("=== 날씨 데이터 테스트 시작 ===")
    db = next(get_db())
    try:
        base_date, base_time, tm = get_base_datetime()
        print(f"기준 시간: {base_date} {base_time} {tm}")
        await fetch_data_weather(db)
        print("날씨 데이터 수집 성공")
    except Exception as e:
        print(f"날씨 데이터 수집 실패: {str(e)}")

async def test_traffic():
    print("=== 교통 데이터 테스트 시작 ===")
    db = next(get_db())
    try:
        await fetch_data_traffic(db)
        print("교통 데이터 수집 성공")
    except Exception as e:
        print(f"교통 데이터 수집 실패: {str(e)}")

async def run_all_tests():
    await test_weather()
    print("\n")
    await test_traffic()

if __name__ == "__main__":
    # asyncio.run(run_all_tests())
    asyncio.run(test_weather())
    # asyncio.run(test_traffic())