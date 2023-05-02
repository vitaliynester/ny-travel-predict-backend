from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import services
from models import RequestModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/predict")
async def predict(payload: RequestModel):
    df = services.create_dataframe(payload)
    result = services.predict(df)
    return {"total_seconds": result}
