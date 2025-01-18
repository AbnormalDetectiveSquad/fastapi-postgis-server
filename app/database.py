from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker 
import os
import platform
from dotenv import load_dotenv

load_dotenv()

SYSTEM = platform.system()

if SYSTEM == "Linux":
    DATABASE_URL = "postgresql://postgres:abcd@localhost:5432/test_model"
else:
    DATABASE_URL = os.getenv("DATABASE_URL")

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
    print(DATABASE_URL)
    try:
        yield db
    finally:
        db.close()