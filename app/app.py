from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, LineString, mapping
from datetime import datetime
from typing import Optional
import models, schemas
from database import get_db
from scheduler import lifespan_scheduler, init_scheduler
import pandas as df

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
@ app.post("/links/", response_model=schemas.LinkNodeNetwork)
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



@app.get("/prediction/{target_tm}")
def load_prediction_data(target_tm: str, db: Session = Depends(get_db)):
    try:
        traffic_prediction = db.query(models.TrafficPrediction.tm,
                models.TrafficPrediction.link_id,
                models.TrafficPrediction.tm,
                models.TrafficPrediction.prediction_5min,
                models.TrafficPrediction.prediction_10min,
                models.TrafficPrediction.prediction_15min)\
            .filter(models.TrafficPrediction.tm == target_tm)\
            .all()
        link_inform=db.query(models.linkidsortorder.start_longitude,
                models.linkidsortorder.start_latitude,
                models.linkidsortorder.end_longitude,
                models.linkidsortorder.end_latitude,
                models.linkidsortorder.middle_longitude,
                models.linkidsortorder.middle_latitude,
                models.linkidsortorder.year_avg_velocity)
        
        traffic_prediction = df.DataFrame(traffic_prediction,columns=['link_id','prediction_5min','prediction_10min','prediction_15min'])
        link_inform= df.DataFrame(link_inform,column=['start_longitude','start_latitude','end_longitude','end_latitude','middle_longitude','middle_latitude','year_avg_velocity'])
        if traffic_prediction is None or link_inform is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        # 응답 데이터 직접 변환
        response_data = {
            "link_id": traffic_prediction['link_id'],
            "5 min": traffic_prediction['prediction_5min'],
            "10 min": traffic_prediction['prediction_10min'],
            "15 min": traffic_prediction['prediction_15min'],
            "start_longitude": link_inform['start_longitude'],
            "start_latitude": link_inform['start_latitude'],
            "end_longitude": link_inform['end_longitude'],
            "end_latitude": link_inform['end_latitude'],
            "middle_longitude": link_inform['middle_longitude'],
            "middle_latitude": link_inform['middle_latitude'],
            "year_avg_velocity": link_inform['year_avg_velocity']  
        }
        return response_data
    except Exception as e:
        raise HTTPException(status_code=404, detail="Prediction not found") from e








