# scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from contextlib import asynccontextmanager
from database import get_db
from fetch import fetch_data_weather, fetch_data_traffic
from predict.main import predict_traffic
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

class ModelSchedulerStatus:
    is_running = False

scheduler = AsyncIOScheduler()
model_status = ModelSchedulerStatus()
def job_listener(event):
    if event.code == EVENT_JOB_EXECUTED:
        logger.info(f"Job executed successfully: {event.job_id}")
    elif event.code == EVENT_JOB_ERROR:
        logger.error(f"Job failed: {event.job_id}")
        logger.error(f"Exception: {event.exception}")

def init_scheduler():
    # 기상데이터: 매시간 40분에 실행
    scheduler.add_job(
        fetch_data_weather,
        trigger=CronTrigger(minute=40),
        args=[next(get_db())],
        misfire_grace_time=None,
        id="fetch_weather"
    )

    # 교통데이터: 매시 0, 5, ... 55분에 실행
    scheduler.add_job(
        fetch_data_traffic,
        trigger=CronTrigger(minute='*/5'),
        args=[next(get_db())],
        id="fetch_traffic"
    )

    # 교통예측모델: 매시 0, 5, ... 55분에 실행
    scheduler.add_job(
        predict_traffic,
        trigger=CronTrigger(minute='*/5'),
        args=[next(get_db()), False, None],
        id="predict_traffic"
    )
    scheduler.pause_job('predict_traffic')  # 초기에는 일시중지 상태로 설정

    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

# 예측 모델 ON/OFF를 위한 함수들 추가
def start_prediction_model():
    if not model_status.is_running:
        scheduler.resume_job('predict_traffic')
        model_status.is_running = True
        logger.info("Traffic prediction model started")
        return {"status": "success", "message": "Prediction model started"}
    return {"status": "info", "message": "Prediction model is already running"}

def stop_prediction_model():
    if model_status.is_running:
        scheduler.pause_job('predict_traffic')
        model_status.is_running = False
        logger.info("Traffic prediction model stopped")
        return {"status": "success", "message": "Prediction model stopped"}
    return {"status": "info", "message": "Prediction model is already stopped"}

def get_prediction_model_status():
    return {
        "is_running": model_status.is_running,
        "next_run_time": scheduler.get_job('predict_traffic').next_run_time
    }

@asynccontextmanager
async def lifespan_scheduler(app):
    scheduler.start()
    yield
    scheduler.shutdown()