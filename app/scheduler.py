# scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from contextlib import asynccontextmanager
from database import get_db
from fetch import fetch_data_weather, fetch_data_traffic
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

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

    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)


@asynccontextmanager
async def lifespan_scheduler(app):
    scheduler.start()
    yield
    scheduler.shutdown()