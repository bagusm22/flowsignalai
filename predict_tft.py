import pandas as pd
import numpy as np
import yfinance as yf

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_lightning import Trainer
from torchmetrics import MeanAbsoluteError

# ==========================
# 1. util teknikal
# ==========================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """pastikan kolom tunggal, bukan multiindex"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        # kadang kolomnya tuple
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


# ==========================
# 2. load data
# ==========================
def load_data(symbol="TLKM.JK", period="2y"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True)
    df = flatten_columns(df)
    df = df.reset_index().rename(columns={"Date": "date"})
    df["symbol"] = symbol

    # time index numerik (wajib buat TFT)
    df["time_idx"] = np.arange(len(df))

    # fitur
    df["return"] = df["Close"].pct_change()
    df["rsi"] = compute_rsi(df["Close"])
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # target = return besok
    df["target"] = df["Close"].shift(-1) / df["Close"] - 1

    df = df.dropna().reset_index(drop=True)
    return df


# ==========================
# 3. dataset TFT
# ==========================
def make_tft_dataset(df: pd.DataFrame):
    # NOTE: kolom tidak boleh punya titik atau multiindex
    df = df.copy()
    max_encoder_length = 60
    max_prediction_length = 1

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["symbol"],
        min_encoder_length=30,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "Close",
            "Volume",
            "return",
            "rsi",
            "vol_ratio",
            "target",
        ],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


# ==========================
# 4. train TFT
# ==========================
def train_tft(dataset: TimeSeriesDataSet):
    train_loader = dataset.to_dataloader(train=True, batch_size=64)
    val_loader = dataset.to_dataloader(train=False, batch_size=64)

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=MeanAbsoluteError(),
        output_size=1,
        reduce_on_plateau_patience=4,
    )

    # ✅ Lightning sudah handle, gak perlu to_lightning_module()
    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        enable_progress_bar=False,  # biar gak spam
        enable_checkpointing=False,  # biar gak warning "loss already saved"
        logger=False,                # matikan tensorboard
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model



# ==========================
# 5. predict (ambil yg paling baru)
# ==========================
def predict_latest(model, dataset: TimeSeriesDataSet):
    # kita ambil satu batch terakhir
    loader = dataset.to_dataloader(train=False, batch_size=64)
    raw_pred, x = model.predict(loader, mode="raw", return_x=True)
    # ambil prediksi paling akhir
    preds = raw_pred.output.detach().cpu().numpy().flatten()
    return preds[-5:]  # misal liat 5 terakhir


if __name__ == "__main__":
    df = load_data("TLKM.JK", "2y")
    dataset = make_tft_dataset(df)
    model = train_tft(dataset)
    preds = predict_latest(model, dataset)
    print("Last predictions:", preds)
