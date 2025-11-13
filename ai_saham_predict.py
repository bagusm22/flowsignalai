import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================
# 0. HELPER
# ==============================

def get_col_series(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s

def to_scalar(x):
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)

def last_trading_close(df):
    df = df.sort_index()
    if "Volume" in df.columns:
        valid = df[df["Volume"] > 0]
        if not valid.empty:
            return float(get_col_series(valid, "Close").iloc[-1])
    return float(get_col_series(df, "Close").iloc[-1])

# ==============================
# 1. INDIKATOR (punyamu yg atas)
# ==============================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_atr_like(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def compute_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    pos_flow, neg_flow = [0], [0]
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            pos_flow.append(money_flow.iloc[i])
            neg_flow.append(0)
        else:
            pos_flow.append(0)
            neg_flow.append(money_flow.iloc[i])
    pos_mf = pd.Series(pos_flow).rolling(period).sum()
    neg_mf = pd.Series(neg_flow).rolling(period).sum()
    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    return mfi.set_axis(close.index)

def compute_adl(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    return mfv.cumsum()

# ==============================
# 2. DOWNLOAD & FEATURE
# ==============================

def get_daily(symbol="ADMR.JK", period="2y"):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    df = df.tz_localize(None)
    return df

def make_features(df):
    df = df.copy()
    close = get_col_series(df, "Close")
    high = get_col_series(df, "High")
    low = get_col_series(df, "Low")
    vol = get_col_series(df, "Volume")

    # PRICE
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # VOLUME
    df["vol_ma_5"] = vol.rolling(5).mean()
    df["vol_ma_20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / vol.rolling(10).mean()
    df["obv"] = compute_obv(close, vol)
    df["mfi"] = compute_mfi(high, low, close, vol, 14)
    df["adl"] = compute_adl(high, low, close, vol)

    # VOLUME SPIKE
    df["vol_spike"] = (vol / df["vol_ma_20"])
    df["is_spike"] = (df["vol_spike"] > 3).astype(int)

    # MOMENTUM
    df["rsi_14"] = compute_rsi(close, 14)
    macd, sig, hist = compute_macd(close)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd, sig, hist
    df["atr_14"] = compute_atr_like(high, low, close, 14)
    df["volatility_10"] = close.pct_change().rolling(10).std()

    # TARGET
    df["close_tomorrow"] = close.shift(-1)
    df["ret_tomorrow"] = (df["close_tomorrow"] / close) - 1.0
    df["target_up"] = (df["ret_tomorrow"] > 0.001).astype(int)

    return df.dropna()

# ==============================
# 3. TRAIN PER SAHAM
# ==============================

def train_xgboost(df):
    feature_cols = [
        "ret_1", "ret_3", "ret_5",
        "ma_5", "ma_20", "ma_ratio",
        "vol_ratio", "vol_spike", "is_spike",
        "obv", "mfi", "adl",
        "rsi_14", "macd", "macd_sig", "macd_hist",
        "atr_14", "volatility_10",
    ]

    X = df[feature_cols]
    y_ret = df["ret_tomorrow"]
    y_cls = df["target_up"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_ret_train, y_ret_test = y_ret.iloc[:split], y_ret.iloc[split:]
    y_cls_train, y_cls_test = y_cls.iloc[:split], y_cls.iloc[split:]

    reg = XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    reg.fit(X_train, y_ret_train)

    cls = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    cls.fit(X_train, y_cls_train)

    # ini boleh kamu matiin nanti kalau udah stabil
    print("=== REGRESI RETURN (XGBoost) ===")
    print("MAE:", mean_absolute_error(y_ret_test, reg.predict(X_test)))
    print("R2 :", r2_score(y_ret_test, reg.predict(X_test)))

    print("\n=== KLASIFIKASI (Naik/Turun) ===")
    print(classification_report(y_cls_test, cls.predict(X_test), digits=4))

    return reg, cls, feature_cols

# ==============================
# 4. KONFIRMASI & FINALIZER (punyamu)
# ==============================

def confirm_signal_from_indicators(last_row, prob_up, ret_pred):
    reasons = []
    score = 0
    max_score = 0

    hard_cap = None
    if prob_up < 0.55:
        hard_cap = "⚠️ LOW"
        reasons.append(f"AI prob rendah ({prob_up:.2%}) → cap ke LOW")

    # AI prob
    max_score += 1
    if prob_up >= 0.65:
        score += 1
        reasons.append("AI prob tinggi (>= 65%)")
    else:
        reasons.append("AI prob rendah (< 65%)")

    # prediksi return
    max_score += 1
    if ret_pred > 0:
        score += 1
        reasons.append("Model prediksi naik")
    else:
        reasons.append("Model prediksi turun")

    # RSI
    rsi_val = to_scalar(last_row["rsi_14"])
    max_score += 1
    rsi_ok = rsi_val >= 40
    if rsi_ok:
        score += 1
        reasons.append(f"RSI OK ({rsi_val:.1f} ≥ 40)")
    else:
        reasons.append(f"RSI lemah ({rsi_val:.1f} < 40)")

    # MACD
    macd_val = to_scalar(last_row["macd"])
    macd_sig = to_scalar(last_row["macd_sig"])
    macd_hist = to_scalar(last_row["macd_hist"])
    max_score += 1
    macd_ok = (macd_val > macd_sig) or (macd_hist > 0)
    if macd_ok:
        score += 1
        reasons.append("MACD mendukung / siap cross")
    else:
        reasons.append("MACD belum konfirmasi")

    # Volume
    vol_ratio = to_scalar(last_row["vol_ratio"])
    max_score += 1
    vol_ok = vol_ratio >= 1.0
    if vol_ok:
        score += 1
        reasons.append(f"Volume OK (ratio {vol_ratio:.2f})")
    else:
        reasons.append(f"Volume rendah (ratio {vol_ratio:.2f})")

    confidence = (score / max_score) * 100

    if confidence >= 80:
        level = "✅ STRONG"
    elif confidence >= 60:
        level = "👍 OK"
    elif confidence >= 40:
        level = "⚠️ LOW"
    else:
        level = "❌ WEAK"

    if hard_cap is not None:
        level = hard_cap
        confidence = min(confidence, 55)

    return {
        "score": score,
        "max_score": max_score,
        "confidence_pct": confidence,
        "level": level,
        "rsi_ok": rsi_ok,
        "macd_ok": macd_ok,
        "vol_ok": vol_ok,
        "vol_ratio": vol_ratio,
        "reasons": reasons,
    }

def make_final_signal(prob_up, ret_pred, confirm, today_price, pred_price, atr):
    model_conf = prob_up
    confirm_conf = confirm["confidence_pct"] / 100

    penalty = 1.0
    if not confirm["vol_ok"]:
        penalty = min(penalty, 0.85)
    if not confirm["macd_ok"]:
        penalty = min(penalty, 0.80)

    base_conf = min(model_conf, confirm_conf)
    final_conf = round(base_conf * penalty, 2)

    # soft rule
    if model_conf >= 0.9 and final_conf < 0.5:
        final_conf = 0.55

    if final_conf >= 0.8:
        final_signal = "BUY_STRONG"
    elif final_conf >= 0.6:
        final_signal = "BUY_WAIT_VOLUME"
    elif final_conf >= 0.5 and ret_pred > 0:
        final_signal = "WATCHLIST"
    else:
        final_signal = "NO_TRADE"

    entry_low = today_price - (0.5 * atr)
    entry_high = today_price
    target_1 = pred_price
    target_2 = pred_price + (0.5 * atr)
    stop_loss = entry_low - atr

    trade_plan = {
        "entry": (round(entry_low, 2), round(entry_high, 2)),
        "targets": (round(target_1, 2), round(target_2, 2)),
        "stop_loss": round(stop_loss, 2),
    }

    return final_signal, final_conf, trade_plan

# ==============================
# 5. PREDIKSI PER SAHAM (SUDAH DIBENERIN)
# ==============================

def predict_tomorrow(symbol, period="10y"):
    """
    Bedanya dengan versi kamu sebelumnya:
    - train dan prediksi dari dataset yg SAMA
    - gak ada download kedua (120d)
    - jadi hasilnya bakal konsisten sama screener yg juga train per saham
    """
    df = get_daily(symbol, period=period)
    df = df[df["Volume"] > 0]

    feat = make_features(df)
    if feat.empty:
        print(f"[{symbol}] ❌ Gagal prediksi: fitur kosong.")
        return

    # train KHUSUS SAHAM INI
    reg, cls, feature_cols = train_xgboost(feat)

    # ambil baris terakhir dari dataset yg barusan dilatih
    last = feat.iloc[-1]
    X_live = last[feature_cols].values.reshape(1, -1)

    # prediksi
    ret_pred = float(reg.predict(X_live)[0])
    prob_up = float(cls.predict_proba(X_live)[0][1])

    today = last_trading_close(df)
    pred_price = today * (1 + ret_pred)

    confirm = confirm_signal_from_indicators(last, prob_up, ret_pred)
    atr_val = to_scalar(last["atr_14"])

    final_signal, final_conf, trade_plan = make_final_signal(
        prob_up=prob_up,
        ret_pred=ret_pred,
        confirm=confirm,
        today_price=today,
        pred_price=pred_price,
        atr=atr_val,
    )

    print(f"\n=== {symbol} — FINAL AI SIGNAL ===")
    print(f"Signal         : {final_signal}")
    print(f"Confidence     : {final_conf:.2f}")
    print(f"Entry ideal    : {trade_plan['entry'][0]} – {trade_plan['entry'][1]}")
    print(f"Target jual    : {trade_plan['targets'][0]} – {trade_plan['targets'][1]}")
    print(f"Stop loss      : {trade_plan['stop_loss']}")

    print("\n--- DETAIL PREDIKSI ---")
    print(f"Harga hari ini : {today:.2f}")
    print(f"Prediksi return: {ret_pred*100:.3f}%")
    print(f"Prediksi harga : {pred_price:.2f}")
    print(f"Prob. naik     : {prob_up:.2%}")
    print(f"Volume ratio   : {confirm['vol_ratio']:.2f}x")

    print("\n--- AI CONFIRMATION ---")
    print(f"Level          : {confirm['level']} ({confirm['confidence_pct']:.1f}%)")
    print(f"Skor           : {confirm['score']}/{confirm['max_score']}")
    for r in confirm["reasons"]:
        print(f"- {r}")

    return {
        "symbol": symbol,
        "signal": final_signal,
        "confidence": final_conf,
        "core": {
            "today": today,
            "pred_return": ret_pred,
            "pred_price": pred_price,
            "prob_up": prob_up,
        },
        "confirmation": confirm,
        "trade_plan": trade_plan,
    }

# ==============================
# 6. MAIN
# ==============================

if __name__ == "__main__":
    predict_tomorrow("MAPI.JK", period="10y")
