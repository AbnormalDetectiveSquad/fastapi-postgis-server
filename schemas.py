from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum
from geoalchemy2.shape import to_shape
from shapely.geometry import mapping

def convert_geometry(obj):
    """GeoAlchemy2 객체를 GeoJSON으로 변환하는 함수"""
    if hasattr(obj, 'point_info') and obj.point_info is not None:
        obj.point_info = mapping(to_shape(obj.point_info))
    if hasattr(obj, 'line_info') and obj.line_info is not None:
        obj.line_info = mapping(to_shape(obj.line_info))
    return obj

#-- Location Master
class LocationType(str, Enum):
    NODE = 'NODE'
    LINK = 'LINK'
    STATION = 'STATION'

class LocationBase(BaseModel):
    location_id: str
    type: LocationType
    name: str
    point_info: Optional[Dict[str, Any]] = None
    line_info: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class LocationCreate(LocationBase):
    pass

class Location(LocationBase):
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

#-- Link Node Network
class LinkNodeNetworkBase(BaseModel):
    link_id: str
    start_node_id: str
    end_node_id: str
    link_name: str
    start_node_name: str
    end_node_name: str
    link_length: float
    speed_limit: Optional[int] = None

class LinkNodeNetworkCreate(LinkNodeNetworkBase):
    pass

class LinkNodeNetwork(LinkNodeNetworkBase):
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

#-- ITS Traffic Data
class ItsTrafficDataBase(BaseModel):
    tm: datetime
    link_id: str
    road_authority: Optional[int] = None
    speed: Optional[Decimal] = None
    travel_time: Optional[int] = None

class ItsTrafficDataCreate(ItsTrafficDataBase):
    pass

class ItsTrafficData(ItsTrafficDataBase):
    created_at: datetime

    class Config:
        from_attributes = True


