import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import warnings

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 0. HELPER
# =========================================================

def get_col_series(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s

def to_scalar(x):
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)

def get_daily_fallback(symbol: str, period: str = "30d"):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.tz_localize(None)
    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]
    return df


# =========================================================
# 1. INDIKATOR SCALPING (NUMPY-SAFE)
# =========================================================

def compute_ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_vwap_safe(df: pd.DataFrame) -> pd.Series:
    """
    VWAP versi aman: semua pakai numpy, baru dibikin Series lagi.
    Ini supaya gak ada error 'Operands are not aligned'.
    """
    high = get_col_series(df, "High").to_numpy(dtype=float)
    low = get_col_series(df, "Low").to_numpy(dtype=float)
    close = get_col_series(df, "Close").to_numpy(dtype=float)
    vol = get_col_series(df, "Volume").to_numpy(dtype=float)

    tp = (high + low + close) / 3.0
    pv_cum = np.cumsum(tp * vol)
    vol_cum = np.cumsum(vol)
    vwap_np = pv_cum / (vol_cum + 1e-9)

    vwap = pd.Series(vwap_np, index=df.index, name="vwap")
    return vwap

# =========================================================
# 2. DOWNLOAD INTRADAY
# =========================================================
def get_intraday_fallback(symbol: str):
    """
    Coba ambil intraday IDX dengan beberapa interval.
    Balikannya: df, period, interval
    """
    candidates = [
        ("5d", "5m"),
        ("10d", "15m"),
        ("30d", "30m"),
        ("60d", "60m"),
    ]
    for period, interval in candidates:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            continue
        df = df.tz_localize(None)
        if "Volume" in df.columns:
            df = df[df["Volume"] > 0]
        if len(df) >= 25:         # minimal biar bisa bikin fitur
            return df, period, interval
    # kalau semua kosong
    return pd.DataFrame(), None, None

def get_intraday(symbol="TLKM.JK", period="5d", interval="5m"):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return df
    df = df.tz_localize(None)
    # buang bar volume 0
    if "Volume" in df.columns:
        df = df[df["Volume"] > 0]
    return df

# =========================================================
# 3. FEATURE ENGINEERING (SCALP)
# =========================================================

def make_features_scalp(df: pd.DataFrame):
    df = df.copy()

    close = get_col_series(df, "Close")
    vol = get_col_series(df, "Volume")

    # return pendek
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)

    # volume pressure
    vol_ma_5 = vol.rolling(5).mean()
    vol_ma_20 = vol.rolling(20).mean()
    df["vol_ma_5"] = vol_ma_5
    df["vol_ma_20"] = vol_ma_20
    df["vol_ratio_5"] = vol / (vol_ma_5 + 1e-9)
    df["vol_ratio_20"] = vol / (vol_ma_20 + 1e-9)

    # micro EMA
    df["ema_9"] = compute_ema(close, 9)
    df["ema_21"] = compute_ema(close, 21)
    df["above_ema"] = (df["ema_9"] > df["ema_21"]).astype(int)

    # VWAP aman
    df["vwap"] = compute_vwap_safe(df)
    df["above_vwap"] = (close > df["vwap"]).astype(int)

    # RSI pendek
    df["rsi_14"] = compute_rsi(close, 14)

    # TARGET: next bar naik >= 0.25%
    df["close_next"] = close.shift(-1)
    df["ret_next"] = (df["close_next"] / close) - 1.0
    df["target_up"] = (df["ret_next"] >= 0.0025).astype(int)

    df = df.dropna()
    return df

# =========================================================
# 4. TRAIN XGB (SCALP)
# =========================================================

def train_xgboost_scalp(df: pd.DataFrame, show_metrics=True):
    feature_cols = [
        "ret_1", "ret_3", "ret_5",
        "vol_ratio_5", "vol_ratio_20",
        "above_ema", "above_vwap",
        "rsi_14",
    ]

    X = df[feature_cols]
    y_ret = df["ret_next"]
    y_cls = df["target_up"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_ret_train, y_ret_test = y_ret.iloc[:split], y_ret.iloc[split:]
    y_cls_train, y_cls_test = y_cls.iloc[:split], y_cls.iloc[split:]

    reg = XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    reg.fit(X_train, y_ret_train)

    cls = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    cls.fit(X_train, y_cls_train)

    if show_metrics:
        print("=== SCALPING REGRESI RETURN (XGBoost) ===")
        print("MAE:", mean_absolute_error(y_ret_test, reg.predict(X_test)))
        print("R2 :", r2_score(y_ret_test, reg.predict(X_test)))
        print("\n=== SCALPING KLASIFIKASI (Naik/Turun) ===")
        print(classification_report(y_cls_test, cls.predict(X_test), digits=4))

    return reg, cls, feature_cols

# =========================================================
# 5. KONFIRMASI & FINALIZER
# =========================================================

def confirm_scalp_from_indicators(last_row, prob_up, ret_pred):
    reasons = []
    score = 0
    max_score = 0

    # 1) AI prob
    max_score += 1
    if prob_up >= 0.65:
        score += 1
        reasons.append("AI prob tinggi (>= 0.65)")
    else:
        reasons.append(f"AI prob rendah ({prob_up:.2f})")

    # 2) volume
    max_score += 1
    vol_ratio = to_scalar(last_row["vol_ratio_5"])
    vol_ok = vol_ratio >= 1.5
    if vol_ok:
        score += 1
        reasons.append(f"Volume spike {vol_ratio:.2f}x")
    else:
        reasons.append(f"Volume biasa {vol_ratio:.2f}x")

    # 3) VWAP
    max_score += 1
    above_vwap = bool(last_row["above_vwap"])
    if above_vwap:
        score += 1
        reasons.append("Harga di atas VWAP")
    else:
        reasons.append("Harga di bawah VWAP")

    # 4) EMA micro
    max_score += 1
    above_ema = bool(last_row["above_ema"])
    if above_ema:
        score += 1
        reasons.append("EMA9 > EMA21")
    else:
        reasons.append("EMA9 <= EMA21")

    confidence = (score / max_score) * 100

    if confidence >= 80:
        level = "✅ STRONG"
    elif confidence >= 60:
        level = "👍 OK"
    else:
        level = "⚠️ LOW"

    return {
        "score": score,
        "max_score": max_score,
        "confidence_pct": confidence,
        "level": level,
        "vol_ratio": vol_ratio,
        "vol_ok": vol_ok,
        "above_vwap": above_vwap,
        "above_ema": above_ema,
        "reasons": reasons,
    }

def make_final_signal_scalp(prob_up, ret_pred, confirm, price_now, tp_pct=0.003, sl_pct=0.003):
    base_conf = min(prob_up, confirm["confidence_pct"] / 100)

    if (
        prob_up >= 0.70
        and confirm["vol_ok"]
        and confirm["above_vwap"]
        and confirm["above_ema"]
    ):
        final_signal = "SCALP_BUY"
    elif prob_up >= 0.55 and confirm["above_vwap"]:
        final_signal = "SCALP_WAIT"
    else:
        final_signal = "NO_TRADE"

    entry = price_now
    target = round(entry * (1 + tp_pct), 2)
    stop = round(entry * (1 - sl_pct), 2)

    trade_plan = {
        "entry": (entry, entry),
        "targets": (target, target),
        "stop_loss": stop,
    }

    return final_signal, round(base_conf, 2), trade_plan

# =========================================================
# 6. PREDIKSI SATU SAHAM (SCALP)
# =========================================================

def predict_scalp(symbol, silent=False):
    # 1. coba intraday dulu
    df, used_period, used_interval = get_intraday_fallback(symbol)

    used_source = "intraday"
    if df.empty or len(df) < 25:
        # 2. kalau intraday gak ada → pakai daily
        df = get_daily_fallback(symbol, period="60d")
        used_source = "daily-fallback"
        used_period = "60d"
        used_interval = "1d"

    if df.empty or len(df) < 25:
        if not silent:
            print(f"[{symbol}] ❌ data nggak ada (intraday & daily).")
        return None

    # 3. fitur tetap pakai fungsi scalping (aman, dia cuma pakai OHLCV + rolling)
    feat = make_features_scalp(df)
    if feat.empty or len(feat) < 20:
        if not silent:
            print(f"[{symbol}] ❌ fitur kurang.")
        return None

    reg, cls, feature_cols = train_xgboost_scalp(feat, show_metrics=not silent)

    last = feat.iloc[-1]
    X_live = last[feature_cols].values.reshape(1, -1)

    ret_pred = float(reg.predict(X_live)[0])
    prob_up = float(cls.predict_proba(X_live)[0][1])

    price_now = float(get_col_series(df, "Close").iloc[-1])

    confirm = confirm_scalp_from_indicators(last, prob_up, ret_pred)

    # kalau sumbernya daily, kita kecilin dikit agresivitasnya
    if used_source == "daily-fallback":
        # misal: kurangin confidence sedikit
        prob_up = prob_up * 0.9

    final_signal, final_conf, trade_plan = make_final_signal_scalp(
        prob_up=prob_up,
        ret_pred=ret_pred,
        confirm=confirm,
        price_now=price_now,
        tp_pct=0.003,
        sl_pct=0.003,
    )

    return {
        "symbol": symbol,
        "signal": final_signal,
        "confidence": final_conf,
        "prob_up": prob_up,
        "pred_return": ret_pred,
        "price_now": price_now,
        "volume_ratio": confirm["vol_ratio"],
        "above_vwap": confirm["above_vwap"],
        "above_ema": confirm["above_ema"],
        "trade_plan": trade_plan,
        "used_period": used_period,
        "used_interval": used_interval,
        "used_source": used_source,
    }



# =========================================================
# 7. SCREENER BANYAK SAHAM
# =========================================================

def screen_stocks_scalp(tickers):
    results = []
    print("\n=== AI SCALPING SCREENER (intraday) ===")
    for i, sym in enumerate(tickers, 1):
        try:
            res = predict_scalp(sym, silent=True)
            if res is None:
                print(f"[{i:3}] {sym:10} -> SKIP (no data at all)")
                continue

            results.append(res)
            src = res.get("used_source", "-")
            print(
                f"[{i:3}] {res['symbol']:8} | {res['signal']:12} | "
                f"conf={res['confidence']:.2f} | prob={res['prob_up']:.2%} | "
                f"vol={res['volume_ratio']:.2f}x | VWAP={'✅' if res['above_vwap'] else '❌'} | "
                f"EMA={'✅' if res['above_ema'] else '❌'} | {src} {res['used_interval']}"
            )
        except Exception as e:
            print(f"[{i:3}] {sym:10} -> ERROR: {e}")
            continue

# =========================================================
# 8. MAIN
# =========================================================

if __name__ == "__main__":
    STOCK = ["ITMG.JK"]
    screen_stocks_scalp(STOCK)
