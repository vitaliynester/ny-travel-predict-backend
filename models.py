from pydantic import BaseModel


class RequestModel(BaseModel):
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    month: int
    day_of_month: int
    hour: int
    day_of_week: int
    pickup_weekends: int
