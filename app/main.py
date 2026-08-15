from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router

app = FastAPI(
    title="AQI Predictor API",
    version="1.0.0",
    description="Backend API for AQI Prediction System"
)

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
def root():
    return {
        "message": "AQI Predictor API is running 🚀"
    }