from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, LineString, mapping
from datetime import datetime
from typing import Optional
import models, schemas
from database import get_db
from scheduler import lifespan_scheduler, init_scheduler, start_prediction_model, stop_prediction_model, get_prediction_model_status
from pydantic import BaseModel 
import pandas as pd

app = FastAPI(lifespan=lifespan_scheduler)
init_scheduler()

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

# -- Location Master
@app.post("/locations/", response_model=schemas.Location)
def create_location(location: schemas.LocationCreate, db: Session = Depends(get_db)):
    db_location = models.LocationMaster(
        location_id=location.location_id,
        type=location.type,
        name=location.name
    )
    
    if location.point_info:
        point = Point(location.point_info['coordinates'])
        db_location.point_info = from_shape(point, srid=4326)
    
    if location.line_info:
        line = LineString(location.line_info['coordinates'])
        db_location.line_info = from_shape(line, srid=4326)

    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

@app.get("/locations/{location_id}")
def read_location(location_id: str, db: Session = Depends(get_db)):
    location = db.query(models.LocationMaster).filter(
        models.LocationMaster.location_id == location_id
    ).first()
    
    if location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # 응답 데이터 직접 변환
    response_data = {
        "location_id": location.location_id,
        "type": location.type,
        "name": location.name,
        "created_at": location.created_at,
        "updated_at": location.updated_at,
        "point_info": mapping(to_shape(location.point_info)) if location.point_info else None,
        "line_info": mapping(to_shape(location.line_info)) if location.line_info else None
    }
    
    return response_data

#-- Link Node Network
@app.post("/links/", response_model=schemas.LinkNodeNetwork)
def create_link(link: schemas.LinkNodeNetworkCreate, db: Session = Depends(get_db)):
    db_link = models.LinkNodeNetwork(**link.dict())
    db.add(db_link)
    db.commit()
    db.refresh(db_link)
    return db_link

@app.get("/links/{link_id}", response_model=schemas.LinkNodeNetwork)
def read_link(link_id: str, db: Session = Depends(get_db)):
    link = db.query(models.LinkNodeNetwork).filter(models.LinkNodeNetwork.link_id == link_id).first()
    if link is None:
        raise HTTPException(status_code=404, detail="Link not found")
    return link

@app.get("/links/", response_model=list[schemas.LinkNodeNetwork])
def read_links(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    links = db.query(models.LinkNodeNetwork).offset(skip).limit(limit).all()
    return links


@app.post("/traffic/", response_model=schemas.ItsTrafficData)
def create_traffic_data(traffic_data: schemas.ItsTrafficDataCreate, db: Session = Depends(get_db)):
    db_traffic = models.ItsTrafficData(**traffic_data.dict())
    db.add(db_traffic)
    db.commit()
    db.refresh(db_traffic)
    return db_traffic

@app.get("/traffic/", response_model=list[schemas.ItsTrafficData])
def read_traffic_data(
    start_time: datetime,
    end_time: datetime,
    link_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.ItsTrafficData).filter(
        models.ItsTrafficData.tm.between(start_time, end_time)
    )
    
    if link_id:
        query = query.filter(models.ItsTrafficData.link_id == link_id)
    
    return query.all()

# 예측 모델 제어 엔드포인트 추가
@app.post("/model/start")
def start_model():
    return start_prediction_model()

@app.post("/model/stop")
def stop_model():
    return stop_prediction_model()

@app.get("/model/status")
def model_status():
    return get_prediction_model_status()

@app.get("/prediction/{target_tm}")
def load_prediction_data(target_tm: str, db: Session = Depends(get_db)):
    try:
        target_tm = datetime.strptime(target_tm, '%Y-%m-%d %H:%M:%S')
        traffic_prediction = db.query( models.TrafficPrediction.tm,
                models.TrafficPrediction.link_id,
                models.TrafficPrediction.tm,
                models.TrafficPrediction.prediction_5min,
                models.TrafficPrediction.prediction_10min,
                models.TrafficPrediction.prediction_15min,
                models.TrafficPrediction.created_at)\
               .filter(models.TrafficPrediction.tm == target_tm).all()
        traffic_df = pd.DataFrame([{
          'tm': row.tm,
          'link_id': row.link_id,
          'prediction_5min': float(row.prediction_5min),
          'prediction_10min': float(row.prediction_10min),
          'prediction_15min': float(row.prediction_15min),
          'at': row.created_at
          } for row in traffic_prediction],
        columns=['tm', 'link_id', 'prediction_5min', 'prediction_10min', 'prediction_15min', 'at']) 
        link_inform=db.query(models.linkidsortorder.start_longitude,
                models.linkidsortorder.start_latitude,
                models.linkidsortorder.end_longitude,
                models.linkidsortorder.end_latitude,
                models.linkidsortorder.middle_longitude,
                models.linkidsortorder.middle_latitude,
                models.linkidsortorder.year_avg_velocity,
                models.linkidsortorder.matrix_index)
        link_inform= pd.DataFrame(link_inform,columns=['start_longitude','start_latitude','end_longitude','end_latitude','middle_longitude','middle_latitude','year_avg_velocity','mi'])
        if traffic_prediction is None or link_inform is None:
           raise HTTPException(status_code=404, detail="Prediction not found")
        # 응답 데이터를 위한 리스트
        response_data = []

        # 각 traffic_row에 대해 매칭된 link_row를 찾아 추가
        for _, traffic_row in traffic_df.iterrows():
            # link_inform에서 해당 row 찾기
            matched_link = link_inform.loc[link_inform['mi'] == traffic_row['link_id']]
            if matched_link.empty:
                continue  # 매칭 실패 시 건너뜀
            
            # matched_link에서 첫 번째 행 추출
            link_row = matched_link.iloc[0]

            # 새로운 row 생성
            new_row = {
                "tm": traffic_row["tm"],
                "link_id": traffic_row["link_id"],
                "prediction_5min": traffic_row["prediction_5min"],
                "prediction_10min": traffic_row["prediction_10min"],
                "prediction_15min": traffic_row["prediction_15min"],
                "start_longitude": link_row["start_longitude"],
                "start_latitude": link_row["start_latitude"],
                "end_longitude": link_row["end_longitude"],
                "end_latitude": link_row["end_latitude"],
                "middle_longitude": link_row["middle_longitude"],
                "middle_latitude": link_row["middle_latitude"],
                "year_avg_velocity": link_row["year_avg_velocity"]
            }

            # 응답 리스트에 추가
            response_data.append(new_row)

        # 최종 응답 반환
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")





@app.get("/get_test")
def get_test():
    return {"message": "This is a GET test"}
class InputModel(BaseModel):
    input: str

@app.post("/post_test")
def post_teet(input: InputModel):
    try:
        print(input.input)
        response_data={
            "respons" : input.input
        }
        return response_data
    except Exception as e:
        print(f"No string: {e}")  
        return "This method is a test function that returns the input string as is."  



