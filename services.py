import math
from math import radians, cos, sin, asin, sqrt

import keras
import pandas as pd

from models import RequestModel


def haversine(pickup_longitude: float,
              dropoff_longitude: float,
              pickup_latitude: float,
              dropoff_latitude: float) -> float:
    lon1 = pickup_longitude
    lon2 = dropoff_longitude
    lat1 = pickup_latitude
    lat2 = dropoff_latitude

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    r = 6371

    return c * r


def create_dataframe(model: RequestModel) -> pd.DataFrame:
    d = {
        'vendor_id': 1,
        'passenger_count': model.passenger_count,
        'pickup_longitude': model.pickup_longitude,
        'pickup_latitude': model.pickup_latitude,
        'dropoff_longitude': model.dropoff_longitude,
        'dropoff_latitude': model.dropoff_latitude,
        'long_lag': abs(model.pickup_longitude - model.dropoff_longitude),
        'lalit_lag': abs(model.pickup_latitude - model.dropoff_latitude),
        'month': model.month,
        'day_of_month': model.day_of_month,
        'hour': model.hour,
        'day_of_week': model.day_of_week,
        'pickup_weekends': model.pickup_weekends,
        'distance_geo': haversine(model.pickup_longitude,
                                  model.dropoff_longitude,
                                  model.pickup_latitude,
                                  model.dropoff_latitude)
    }
    df = pd.DataFrame(data=d, index=[1])
    return df


def predict(df: pd.DataFrame) -> int:
    model = keras.models.load_model("model.hdf5", compile=False)
    pred = model.predict(df, verbose=False)
    return int(math.exp(pred[0]) - 1)
