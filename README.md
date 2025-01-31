# 교통 예측 시스템

## 개요
이 시스템은 FastAPI 기반의 교통 예측 시스템으로, 실시간 교통 데이터와 기상 데이터를 수집하고 ST-GCN(시공간 그래프 합성곱 신경망) 모델을 통해 미래 교통 상황을 예측합니다.

## 주요 기능
- 실시간 데이터 수집 및 저장
  - 교통 데이터 수집
  - 기상 데이터 수집
- 교통 상황 예측
  - 5분/10분/15분 후 예측
  - ST-GCN 모델 기반 예측
  - 예측 결과 데이터베이스 저장
- API 서비스
  - 예측 모델 상태 관리 (시작/중지)
  - 위치 정보 관리 (노드/링크/스테이션)
  - 예측 결과 조회

## 시스템 구조

### 디렉토리 구조
```
app/
├── app.py              # FastAPI 메인 애플리케이션
├── database.py         # 데이터베이스 연결 및 설정
├── fetch.py           # 외부 API 데이터 수집
├── models.py          # 데이터베이스 모델 정의
├── schemas.py         # Pydantic 스키마 정의
├── scheduler.py       # 데이터 수집 및 예측 스케줄러
└── predict/           # 예측 모델 관련 코드
    ├── main.py        # 예측 프로세스 메인 로직
    ├── data/          # 모델 가중치 및 그래프 구조 데이터
    │   ├── weight.pt  # 학습된 모델 가중치
    │   └── adj_matrix.npz # 도로 네트워크 인접 행렬
    └── model/         # ST-GCN 모델 구현 코드
```

### API 엔드포인트
- 위치 정보
  - `GET /locations/{location_id}`: 특정 위치 정보 조회
  - `POST /links/`: 새로운 링크 생성
  - `GET /links/{link_id}`: 특정 링크 정보 조회
- 예측 모델 제어
  - `POST /model/start`: 예측 모델 시작
  - `POST /model/stop`: 예측 모델 중지
  - `GET /model/status`: 예측 모델 상태 확인
  - `GET /prediction/{target_tm}`: 특정 시점의 예측 결과 조회

## 예측 모델 (ST-GCN)

### 예측 프로세스
1. 최근 교통 데이터 수집
2. 기상 데이터 통합
3. 공휴일/요일 정보 처리
4. 데이터 정규화 및 텐서 변환
5. 모델 예측 수행
6. 결과 후처리 및 저장

### 주요 모델 컴포넌트
- `STGCNChebGraphConv_OSA`: 메인 모델 클래스
- `ChebGraphConv`: Chebyshev 그래프 합성곱 레이어
- `TemporalConvLayer`: 시간적 특성 추출 레이어
- `OutputBlock_OSA`: OSA 구조가 적용된 출력 블록

## 환경 설정
데이터베이스 연결을 위한 환경 변수:
- `DB_HOST`: 데이터베이스 호스트
- `DB_PORT`: 데이터베이스 포트
- `DB_NAME`: 데이터베이스 이름
- `DB_USER`: 데이터베이스 사용자
- `DB_PASSWORD`: 데이터베이스 비밀번호