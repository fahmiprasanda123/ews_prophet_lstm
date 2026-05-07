"""
Forecast API router for Agri-AI EWS.
"""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
import pandas as pd

from api.schemas import (
    ForecastResponse, ForecastPoint, EWSStatusResponse, EWSAlert, 
    ModelComparisonResponse, ModelMetrics
)
from data.database import get_store

router = APIRouter(prefix="/api", tags=["Forecast"])


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    province: str = Query(..., description="Province name"),
    commodity: str = Query(..., description="Commodity name"),
    days: int = Query(30, ge=1, le=120, description="Forecast days"),
    model: str = Query("prophet", description="Model: prophet, lstm, hybrid"),
):
    """Generate price forecast for a specific province and commodity."""
    store = get_store()
    df = store.load_all()

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    series = df[(df['province'] == province) & (df['commodity'] == commodity)]
    if series.empty:
        raise HTTPException(status_code=404, detail=f"No data for {commodity} in {province}")

    current_price = series['price'].iloc[-1]

    try:
        if model in ("prophet", "hybrid"):
            from models.prophet_forecast import FoodPriceProphet
            p = FoodPriceProphet(df)
            forecast = p.train_and_forecast(province, commodity, periods=days)
            
            points = []
            future_rows = forecast[forecast['ds'] > series['date'].max()]
            for _, row in future_rows.iterrows():
                points.append(ForecastPoint(
                    date=row['ds'].strftime('%Y-%m-%d'),
                    predicted_price=round(float(row['yhat']), 2),
                    lower_bound=round(float(row['yhat_lower']), 2),
                    upper_bound=round(float(row['yhat_upper']), 2),
                ))

            return ForecastResponse(
                province=province, commodity=commodity,
                model_used=model, current_price=float(current_price),
                forecast=points[:days],
            )

        elif model == "lstm":
            from models.lstm_forecast import LSTMForecaster
            l = LSTMForecaster(seq_length=30)
            X, y = l.prepare_data(df, province, commodity)
            l.train_single_series(X[-300:], y[-300:], epochs=5)
            last_30 = series['price'].values[-30:]
            preds = l.predict_multi_step(last_30, steps=days)

            last_date = series['date'].max()
            points = [
                ForecastPoint(
                    date=(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    predicted_price=round(float(p), 2),
                )
                for i, p in enumerate(preds)
            ]

            return ForecastResponse(
                province=province, commodity=commodity,
                model_used="lstm", current_price=float(current_price),
                forecast=points,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@router.get("/ews/status", response_model=EWSStatusResponse)
def get_ews_status(
    commodity: str = Query(None, description="Filter by commodity"),
):
    """Get current EWS status for all provinces (or filtered)."""
    store = get_store()
    df = store.load_all()

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    from engine.ews_engine_v2 import EWSEngineV2
    ews = EWSEngineV2(df)

    commodities = [commodity] if commodity else df['commodity'].unique().tolist()
    provinces = df['province'].unique().tolist()

    alerts = []
    for comm in commodities:
        for prov in provinces:
            series = df[(df['province'] == prov) & (df['commodity'] == comm)]
            if series.empty:
                continue
            current = series['price'].iloc[-1]
            # Simple prediction: use 7-day trend projection
            if len(series) >= 7:
                trend = (series['price'].iloc[-1] - series['price'].iloc[-7]) / series['price'].iloc[-7]
                predicted = current * (1 + trend)
            else:
                predicted = current

            result = ews.calculate_composite_score(prov, comm, predicted)
            if result['level'] in ('Danger', 'Alert', 'Watch'):
                alerts.append(EWSAlert(
                    province=prov, commodity=comm,
                    level=result['level'], score=result['score'],
                    message=result['message'], factors=result['factors'],
                    recommendations=result['recommendations'],
                ))

    return EWSStatusResponse(
        alerts=alerts,
        timestamp=datetime.now().isoformat(),
        total_danger=sum(1 for a in alerts if a.level == 'Danger'),
        total_alert=sum(1 for a in alerts if a.level == 'Alert'),
    )


@router.get("/models/compare", response_model=ModelComparisonResponse)
def compare_models(
    province: str = Query(...),
    commodity: str = Query(...),
):
    """Compare Prophet vs LSTM model performance."""
    store = get_store()
    df = store.load_all()

    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")

    from models.evaluation import calculate_metrics

    results = []

    # Prophet evaluation
    try:
        from models.prophet_forecast import FoodPriceProphet
        from prophet import Prophet
        fp = FoodPriceProphet(df)
        p_df = fp.prepare_data(province, commodity)
        train_df, test_df = fp.split_data(p_df, test_size=0.2)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(train_df)
        pred = m.predict(test_df[['ds']])
        metrics = calculate_metrics(test_df['y'].values, pred['yhat'].values, "Prophet")
        results.append(ModelMetrics(
            model_name="Prophet", rmse=metrics['RMSE'], mae=metrics['MAE'],
            mape=metrics['MAPE (%)'], r2=metrics.get('R²'),
            smape=metrics.get('SMAPE (%)'),
            directional_accuracy=metrics.get('Directional Accuracy (%)'),
        ))
    except Exception as e:
        pass

    # LSTM evaluation
    try:
        from models.lstm_forecast import LSTMForecaster
        import torch
        lf = LSTMForecaster(seq_length=30)
        X, y = lf.prepare_data(df, province, commodity)
        Xtr, Xte, ytr, yte = lf.split_data(X, y, test_size=0.2)
        lf.train_single_series(Xtr, ytr, epochs=5)
        lf.model.eval()
        with torch.no_grad():
            yp = lf.model(Xte)
            y_pred = lf.scaler.inverse_transform(yp.numpy().reshape(-1, 1))
            y_true = lf.scaler.inverse_transform(yte.numpy().reshape(-1, 1))
            metrics = calculate_metrics(y_true, y_pred, "LSTM")
            results.append(ModelMetrics(
                model_name="LSTM", rmse=metrics['RMSE'], mae=metrics['MAE'],
                mape=metrics['MAPE (%)'], r2=metrics.get('R²'),
                smape=metrics.get('SMAPE (%)'),
                directional_accuracy=metrics.get('Directional Accuracy (%)'),
            ))
    except Exception:
        pass

    best = min(results, key=lambda x: x.mape).model_name if results else "None"

    return ModelComparisonResponse(
        province=province, commodity=commodity,
        models=results, best_model=best,
    )
