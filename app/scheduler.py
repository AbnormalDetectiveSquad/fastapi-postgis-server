# scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
from database import get_db
from fetch import fetch_data_weather, fetch_data_traffic

scheduler = AsyncIOScheduler()

def init_scheduler():
    # 기상데이터: 매시간 40분에 실행
    scheduler.add_job(
        fetch_data_weather,
        trigger=CronTrigger(minute=40),
        args=[next(get_db())],
        id="fetch_weather"
    )

    # 교통데이터: 매시 0, 5, ... 55분에 실행
    scheduler.add_job(
        fetch_data_traffic,
        trigger=CronTrigger(minute='*/5'),
        args=[next(get_db())],
        id="fetch_traffic"
    )

@asynccontextmanager
async def lifespan_scheduler(app):
    scheduler.start()
    yield
    scheduler.shutdown()