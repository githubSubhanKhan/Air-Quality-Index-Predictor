from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.routes.history import router as history_router
from app.routes.meta import router as meta_router

app = FastAPI(
    title="AQI Predictor API",
    version="1.0.0",
    description="Backend API for AQI Prediction System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)
app.include_router(meta_router)


@app.get("/")
def root():
    return {
        "message": "AQI Predictor API is running 🚀"
    }