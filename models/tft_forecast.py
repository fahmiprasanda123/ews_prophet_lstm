"""
Temporal Fusion Transformer (TFT) forecaster for Agri-AI EWS.
Provides multi-horizon, interpretable forecasting with attention-based architecture.

Requires: pytorch-forecasting, pytorch-lightning
Falls back gracefully if not installed.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Check if TFT dependencies are available
TFT_AVAILABLE = False
try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    logger.info("pytorch-forecasting not installed. TFT model unavailable.")


class TFTForecaster:
    """Temporal Fusion Transformer based forecaster.
    
    Features:
    - Multi-horizon forecasting (predicts all future steps simultaneously)
    - Interpretable attention weights
    - Handles static (province, commodity) and time-varying (weather) covariates
    - Probabilistic output via quantile loss
    """

    def __init__(self, max_prediction_length=30, max_encoder_length=90):
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.model = None
        self.training_dataset = None
        self._is_available = TFT_AVAILABLE

    @property
    def is_available(self):
        return self._is_available

    def prepare_dataset(self, df, province=None, commodity=None, weather_df=None):
        """Prepare data for TFT training.
        
        Args:
            df: Full DataFrame with date, province, commodity, price.
            province: Optional filter for specific province.
            commodity: Optional filter for specific commodity.
            weather_df: Optional weather features DataFrame.
        
        Returns:
            TimeSeriesDataSet ready for training.
        """
        if not self._is_available:
            raise RuntimeError("pytorch-forecasting is not installed. Install with: pip install pytorch-forecasting pytorch-lightning")

        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        if province:
            data = data[data['province'] == province]
        if commodity:
            data = data[data['commodity'] == commodity]

        # Create time index (integer, required by TFT)
        data = data.sort_values(['province', 'commodity', 'date'])
        data['time_idx'] = data.groupby(['province', 'commodity']).cumcount()
        
        # Create group ID
        data['group_id'] = data['province'] + '_' + data['commodity']
        
        # Add time features
        data['month'] = data['date'].dt.month.astype(str)
        data['day_of_week'] = data['date'].dt.dayofweek.astype(str)
        data['is_wet_season'] = data['date'].dt.month.isin([11, 12, 1, 2, 3, 4]).astype(float)
        
        # Add weather if available
        time_varying_known = ['is_wet_season']
        if weather_df is not None and not weather_df.empty:
            weather_df.index = pd.to_datetime(weather_df.index)
            for col in ['rainfall_mm', 'enso_index']:
                if col in weather_df.columns:
                    data[col] = data['date'].map(weather_df[col].to_dict()).fillna(0).astype(float)
                    time_varying_known.append(col)

        # Ensure no NaN in price
        data = data.dropna(subset=['price'])

        max_time_idx = data['time_idx'].max()
        training_cutoff = max_time_idx - self.max_prediction_length

        self.training_dataset = TimeSeriesDataSet(
            data[data['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target='price',
            group_ids=['group_id'],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=['province', 'commodity'],
            time_varying_known_categoricals=['month', 'day_of_week'],
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=['price'],
            target_normalizer=GroupNormalizer(groups=['group_id']),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        return self.training_dataset, data

    def train(self, dataset=None, max_epochs=15, batch_size=64, learning_rate=0.001,
              hidden_size=32, attention_head_size=2, dropout=0.1, gpus=0):
        """Train the TFT model.
        
        Args:
            dataset: TimeSeriesDataSet (uses self.training_dataset if None).
            max_epochs: Maximum training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            hidden_size: Hidden layer size.
            attention_head_size: Number of attention heads.
            dropout: Dropout rate.
            gpus: Number of GPUs (0 for CPU).
        """
        if not self._is_available:
            raise RuntimeError("pytorch-forecasting not installed.")

        dataset = dataset or self.training_dataset
        if dataset is None:
            raise ValueError("No dataset provided. Call prepare_dataset() first.")

        # Create dataloader
        train_dataloader = dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )

        # In pytorch-forecasting, dataset.data is a dict. We should use the original dataframe.
        # But we only need train_dataloader for a quick train.
        # Let's just use train_dataloader and skip validation to make it faster and error-free for the dashboard.
        # Initialize model
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=16,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if gpus > 0 else "cpu",
            devices=gpus if gpus > 0 else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            enable_progress_bar=True,
        )

        trainer.fit(self.model, train_dataloaders=train_dataloader)
        logger.info("TFT training complete.")

    def predict(self, data, dataset=None):
        """Generate predictions from the trained model.
        
        Returns:
            dict with keys: mean, lower, upper (np.ndarray each)
        """
        if not self._is_available or self.model is None:
            raise RuntimeError("Model not trained or pytorch-forecasting not available.")

        dataset = dataset or self.training_dataset
        if dataset is None:
            raise ValueError("No dataset available for prediction.")

        # Create prediction dataset
        pred_dataset = TimeSeriesDataSet.from_dataset(
            dataset, data, predict=True, stop_randomization=True
        )
        pred_dataloader = pred_dataset.to_dataloader(
            train=False, batch_size=128, num_workers=0
        )

        # Get predictions
        predictions = self.model.predict(pred_dataloader, return_x=True)
        pred_values = predictions.output.numpy()

        # Extract quantiles (median, 5th, 95th percentile)
        if pred_values.ndim == 3:
            return {
                'mean': pred_values[:, :, 3].flatten(),    # Median (q=0.5)
                'lower': pred_values[:, :, 0].flatten(),   # Lower (q=0.02)
                'upper': pred_values[:, :, -1].flatten(),  # Upper (q=0.98)
            }
        else:
            return {
                'mean': pred_values.flatten(),
                'lower': pred_values.flatten() * 0.95,
                'upper': pred_values.flatten() * 1.05,
            }

    def get_variable_importance(self):
        """Extract variable importance scores from the trained model.
        
        Returns:
            dict with importance scores per variable.
        """
        if self.model is None:
            return {}

        try:
            interpretation = self.model.interpret_output(
                self.model.predict(
                    self.training_dataset.to_dataloader(train=False, batch_size=128, num_workers=0),
                    return_x=True
                ),
                reduction='mean'
            )
            return {
                'encoder_importance': interpretation.get('encoder_variables', {}),
                'decoder_importance': interpretation.get('decoder_variables', {}),
                'static_importance': interpretation.get('static_variables', {}),
            }
        except Exception as e:
            logger.warning(f"Could not extract variable importance: {e}")
            return {}


class TFTFallback:
    """Fallback when TFT is not available. Returns None for all operations."""
    
    def __init__(self, *args, **kwargs):
        self._is_available = False

    @property
    def is_available(self):
        return False

    def prepare_dataset(self, *args, **kwargs):
        return None, None

    def train(self, *args, **kwargs):
        logger.warning("TFT not available. Install pytorch-forecasting.")
        return None

    def predict(self, *args, **kwargs):
        return None

    def get_variable_importance(self):
        return {}


def get_tft_forecaster(**kwargs):
    """Factory function that returns TFT or fallback based on availability."""
    if TFT_AVAILABLE:
        return TFTForecaster(**kwargs)
    return TFTFallback(**kwargs)
