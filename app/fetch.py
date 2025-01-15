import httpx
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from urllib.parse import unquote
from models import ItsTrafficData, KmaWeatherData  # DB 모델 임포트

SEOUL_COORDINATES = [(60, 127), (61, 127), (60, 126), (59, 126), (61, 126), (62, 126), (62, 127), (62, 128), (60, 128),
                     (61, 128), (61, 129), (62, 129), (59, 127), (59, 128), (58, 127), (58, 126), (57, 126), (57, 127),
                     (58, 125), (57, 125), (59, 124), (58, 124), (59, 125), (60, 125), (61, 125), (61, 124), (62, 125),
                     (63, 125), (63, 126), (63, 127), ]

# base_date와 base_time을 계산하는 함수
def get_base_datetime():
    now = datetime.now()
    # 현재 시각이 40분 이전이면 이전 시각의 데이터를 가져옴
    if now.minute < 40:
        now = now - timedelta(hours=1)

    base_date = now.strftime("%Y%m%d")
    base_time = now.strftime("%H00")
    tm = now.replace(minute=0, second=0, microsecond=0)

    return base_date, base_time, tm


async def fetch_data_weather(db: Session):
    # API 설정
    API_KEY = unquote('BoIWc771l1d7RuIhnt5NqWR4IZm7FfDM6UhqSCMBN1P%2BaHp9TrzS1bkZmO93GcKiR4zK0qCgkA%2F8EhVMgO9jPA%3D%3D')
    BASE_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'

    base_date, base_time, tm = get_base_datetime()

    async with httpx.AsyncClient() as client:
        try:
            for coordinates in SEOUL_COORDINATES:
                (nx, ny) = coordinates
                # base_date, base_time 관련 로직 추가
                params = {'serviceKey': API_KEY, 'pageNo': '1', 'numOfRows': '1000',
                          'dataType': 'JSON', 'base_date': base_date, 'base_time': base_time,
                          'nx': nx, 'ny': ny}

                # API에서 데이터 가져오기
                response = await client.get(BASE_URL, params=params, timeout=10)
                data = response.json()

                pty = [item for item in data['response']['body']['items']['item']
                       if item['category'] == 'PTY'][0]
                rn1 = [item for item in data['response']['body']['items']['item']
                       if item['category'] == 'RN1'][0]

                pty_value = int(pty['obsrValue'])
                rn1_value = round(float(rn1['obsrValue']), 1)

                new_record = KmaWeatherData(
                    nx=nx, ny=ny, tm=tm,
                    pty=pty_value,
                    rn1=rn1_value
                )

                db.add(new_record)
                db.commit()

        except Exception as e:
            print(f"Error fetching data: {e}")
            db.rollback()


async def fetch_data_traffic(db: Session):
    # API 설정
    API_KEY = '25b9372d7c39424aa49f2c27c47c6276'
    BASE_URL = "https://openapi.its.go.kr:9443/trafficInfo"

    # 서울 위경도 범위 check (현재 강남구 한정)
    params = {
        'apiKey': API_KEY,
        'type': 'all',
        'getType': 'json',
        'minX': 127.01,
        'maxX': 127.13,
        'minY': 37.45,
        'maxY': 37.54,
    }

    async with httpx.AsyncClient() as client:
        try:
            # API에서 데이터 가져오기
            response = await client.get(BASE_URL, params=params, timeout=10)
            data = response.json()

            records_to_insert = []
            for item in data['body']['items']:
                tm = datetime.strptime(item['createdDate'], '%Y%m%d%H%M%S')

                # 데이터 변환 및 매핑
                record = {
                    'tm':tm,
                    'link_id': item['linkId'],
                    'speed':  float(item['speed']),
                    'travel_time': float(item['travelTime'])
                }
                records_to_insert.append(record)

            # db insert
            db.bulk_insert_mappings(ItsTrafficData, records_to_insert)
            db.commit()

        except Exception as e:
            print(f"Error fetching data: {e}")
            db.rollback()

