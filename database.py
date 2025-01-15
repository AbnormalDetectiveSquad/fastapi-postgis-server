from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# 환경변수에서 DB 설정 가져오기
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:vlftnqhdks12#@localhost:5432/ads_db"
)

# 엔진 설정에 추가 옵션
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()