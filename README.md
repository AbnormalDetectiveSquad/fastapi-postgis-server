# FastAPI PostgreSQL API Server

PostgreSQL 데이터베이스와 연동된 FastAPI 기반의 REST API 서버입니다.

## 기능

### Location Master API
- 위치 정보 조회 (`GET /locations/{location_id}`)
- 새로운 위치 정보 생성 (`POST /locations/`)
- PostGIS를 활용한 공간 데이터 처리 (Point, LineString)

### Link Node Network API
- 링크 정보 조회 (`GET /links/{link_id}`)
- 링크 목록 조회 (`GET /links/`)
- 새로운 링크 생성 (`POST /links/`)

### Traffic Data API
- 교통 데이터 조회 (`GET /traffic/`)
  - 시간 범위 기반 필터링
  - 링크 ID 기반 필터링
- 새로운 교통 데이터 추가 (`POST /traffic/`)

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install fastapi sqlalchemy psycopg2-binary geoalchemy2 shapely
```

2. PostgreSQL 데이터베이스 설정:
- RDS 설정... 예정...


3. 환경 변수 설정:
```bash
# .env 파일 생성
DATABASE_URL=postgresql://username:password@localhost:5432/ads_db
```

## 실행 방법

```bash
# FastAPI 서버 실행
uvicorn app:app --reload
```

## API 테스트

테스트 스크립트를 사용하여 각 엔드포인트 테스트:
```python
# Location API 테스트
python test_getapi.py

# 응답 예시
{
    "location_id": "3420004203",
    "type": "NODE",
    "name": "...",
    "point_info": {
        "type": "Point",
        "coordinates": [127.xxx, 37.xxx]
    }
}
```

## 프로젝트 구조

```
.
├── app.py          # FastAPI 애플리케이션 및 라우터
├── models.py       # SQLAlchemy 모델
├── schemas.py      # Pydantic 스키마
├── database.py     # DB 연결 설정
└── test_getapi.py  # API 테스트
```

## API 문서

서버 실행 후 다음 URL에서 API 문서 확인 가능:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)