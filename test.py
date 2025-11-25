import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings, os, traceback

INTERVAL = "5m"

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - kalau kolom MultiIndex / tuple -> ambil level pertama
    - rename ke Title Case
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1) pipihkan multiindex
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for c in df.columns:
            # c bisa tuple seperti ('Open', 'BBCA.JK')
            if isinstance(c, tuple):
                new_cols.append(c[0])
            else:
                new_cols.append(c)
        df.columns = new_cols
    else:
        # kadang bukan MultiIndex tapi isinya tuple juga
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                new_cols.append(c[0])
            else:
                new_cols.append(c)
        df.columns = new_cols

    # 2) samakan kapitalisasi
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    return df

def download_5m(symbol: str, period: str = "5d") -> pd.DataFrame:
    """
    Download data 5 menit, dipaksa group_by=column supaya kolomnya standar:
    Open, High, Low, Close, Adj Close, Volume
    """
    raw = yf.download(
        symbol,
        period=period,
        interval=INTERVAL,
        progress=False,
        auto_adjust=False, # << penting banget
    )
    # beberapa versi yfinance balikin (df, meta)
    if isinstance(raw, tuple):
        df = raw[0]
    else:
        df = raw

    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_df(df)
    
    return df

def download_single(symbol: str, period: str = "3y") -> pd.DataFrame:
    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          progress=False, auto_adjust=False)

        # beberapa versi yfinance balikin (df, meta)
        if isinstance(raw, tuple):
            df = raw[0]
        else:
            df = raw

        if df is None or df.empty:
            return pd.DataFrame()

        df = _normalize_df(df)

        # buang baris volume 0
        if "Volume" in df.columns:
            df = df[df["Volume"] > 0]

        # hapus timezone
        try:
            df = df.tz_localize(None)
        except Exception:
            pass

        return df
    except Exception as e:
        log(f"❌ Error download {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    
    df = download_single("BBCA.JK", "2d")
    # ihsg = download_single("^JKSE", "2d")
    print(df)
    # print(ihsg)