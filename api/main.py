"""
FastAPI main application for Agri-AI EWS REST API.
Provides RESTful endpoints for forecasting, data access, and EWS status.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import forecast, data

app = FastAPI(
    title="Agri-AI EWS API",
    description=(
        "REST API untuk Early Warning System harga pangan Indonesia. "
        "Menyediakan forecast, data historis, dan status peringatan dini."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(forecast.router)
app.include_router(data.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Agri-AI EWS API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    from data.database import get_store
    store = get_store()
    stats = store.get_stats()
    return {
        "status": "healthy",
        "database": {
            "total_records": stats['total_records'],
            "date_range": f"{stats['date_from']} to {stats['date_to']}",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
