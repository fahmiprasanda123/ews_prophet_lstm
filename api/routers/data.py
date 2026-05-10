"""
Data API router for Agri-AI EWS.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime

from api.schemas import PriceRecord, DataStatsResponse, SyncResponse, SupplyRiskResponse
from data.database import get_store

router = APIRouter(prefix="/api/data", tags=["Data"])


@router.get("/prices", response_model=List[PriceRecord])
def get_prices(
    province: str = Query(None),
    commodity: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    limit: int = Query(100, le=5000),
):
    """Get historical price data with optional filters."""
    store = get_store()
    
    if province and commodity:
        df = store.get_series(province, commodity, start_date, end_date)
        df['province'] = province
        df['commodity'] = commodity
    else:
        df = store.load_all()
        if province:
            df = df[df['province'] == province]
        if commodity:
            df = df[df['commodity'] == commodity]
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]

    df = df.tail(limit)
    df['date'] = df['date'].astype(str)
    
    return df[['date', 'province', 'commodity', 'price']].to_dict('records')


@router.get("/commodities")
def get_commodities():
    """List all available commodities."""
    store = get_store()
    return {"commodities": store.get_commodities()}


@router.get("/provinces")
def get_provinces():
    """List all available provinces."""
    store = get_store()
    return {"provinces": store.get_provinces()}


@router.get("/stats", response_model=DataStatsResponse)
def get_stats():
    """Get database statistics."""
    store = get_store()
    stats = store.get_stats()
    return DataStatsResponse(**stats)


@router.get("/latest")
def get_latest_prices(commodity: str = Query(None)):
    """Get the latest price for each province-commodity combo."""
    store = get_store()
    df = store.get_latest_prices(commodity)
    df['date'] = df['date'].astype(str)
    return df.to_dict('records')


@router.post("/sync", response_model=SyncResponse)
def trigger_sync():
    """Manually trigger PIHPS data synchronization."""
    store = get_store()
    try:
        from data.scheduler import DataSyncScheduler
        scheduler = DataSyncScheduler(store)
        count = scheduler.run_once()
        return SyncResponse(
            status="ok",
            records_added=count,
            message=f"Sync complete. {count} records added.",
        )
    except Exception as e:
        return SyncResponse(
            status="error",
            records_added=0,
            message=f"Sync failed: {str(e)}",
        )


@router.get("/supply-risk", response_model=SupplyRiskResponse)
def get_supply_risk(
    province: str = Query(...),
    commodity: str = Query(...),
):
    """Get supply risk score for a province-commodity pair."""
    store = get_store()
    df = store.load_all()

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    from engine.supply_risk import SupplyRiskScorer
    scorer = SupplyRiskScorer(df)
    result = scorer.calculate_risk_score(province, commodity)

    return SupplyRiskResponse(
        province=province, commodity=commodity,
        score=result['score'], factors=result['factors'],
        trend_direction=result['trend_direction'],
        description=result['description'],
    )
