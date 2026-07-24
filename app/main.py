from fastapi import FastAPI
from app.routes.health import router as health_router

app = FastAPI(
    title="AQI Predictor API",
    version="1.0.0",
    description="Backend API for AQI Prediction System"
)

app.include_router(health_router)


@app.get("/")
def root():
    return {
        "message": "AQI Predictor API is running 🚀"
    }