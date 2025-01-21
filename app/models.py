from sqlalchemy import Column, String, Integer, Float, DateTime, BigInteger, Numeric,Enum, CheckConstraint
from geoalchemy2 import Geometry
from sqlalchemy.sql import func
from database import Base
import enum

# ENUM 타입 정의
class LocationType(enum.Enum):
    NODE = 'NODE'
    LINK = 'LINK'
    STATION = 'STATION'

class LocationMaster(Base):
    __tablename__ = 'location_master'
    
    location_id = Column(String(50), primary_key=True)
    type = Column(Enum(LocationType), nullable=False)
    name = Column(String(100), nullable=False)
    point_info = Column(Geometry('POINT', srid=4326))
    line_info = Column(Geometry('LINESTRING', srid=4326))
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("""
            CASE type
                WHEN 'NODE' THEN point_info IS NOT NULL AND line_info IS NULL
                WHEN 'LINK' THEN line_info IS NOT NULL AND point_info IS NULL
                WHEN 'STATION' THEN point_info IS NOT NULL
            END
        """, name='chk_geometry'),
    )


class LinkNodeNetwork(Base):
    __tablename__ = 'link_node_network'

    link_id = Column(String(50), primary_key=True)
    start_node_id = Column(String(50), nullable=False)
    end_node_id = Column(String(50), nullable=False)
    link_name = Column(String(100), nullable=False)
    start_node_name = Column(String(100), nullable=False)
    end_node_name = Column(String(100), nullable=False)
    link_length = Column(Float, nullable=False)
    speed_limit = Column(Integer)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class LinkGridMapping(Base):
    __tablename__ = "link_grid_mapping"

    link_id = Column(String(10), primary_key=True)
    nx = Column(Integer)
    ny = Column(Integer)


class ItsTrafficData(Base):
    __tablename__ = 'its_traffic_data'
    
    tm = Column(DateTime, primary_key=True)
    link_id = Column(String(10), primary_key=True, nullable=False)
    road_authority = Column(Integer)
    speed = Column(Numeric(5,1))
    travel_time = Column(Integer)
    # created_at = Column(DateTime, nullable=False, server_default=func.now())


class KmaWeatherData(Base):
    __tablename__ = 'kma_weather_data'

    nx = Column(Integer, primary_key=True, nullable=False)
    ny = Column(Integer, primary_key=True, nullable=False)
    tm = Column(DateTime, primary_key=True, nullable=False)
    pty = Column(Numeric(5,1))
    rn1 = Column(Numeric(5,1))
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class TrafficPrediction(Base):
    __tablename__ = "traffic_predictions"
    
    link_id = Column(String(10), primary_key=True)
    tm = Column(DateTime, primary_key=True)
    prediction_5min = Column(Numeric(5,1), nullable=False)
    prediction_10min = Column(Numeric(5,1), nullable=False)
    prediction_15min = Column(Numeric(5,1), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class linkidsortorder(Base):
    __tablename__ = 'link_id_sort_order'

    matrix_index = Column(Integer, primary_key=True, nullable=False)
    link_id = Column(String(10), primary_key=True, nullable=False)
    start_longitude = Column(Float)
    start_latitude = Column(Float)
    end_longitude = Column(Float)
    end_latitude = Column(Float)
    middle_longitude = Column(Float)
    middle_latitude = Column(Float)
    year_avg_velocity = Column(Float)