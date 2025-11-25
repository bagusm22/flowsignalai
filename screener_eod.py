import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import warnings

# coba pakai tabulate, kalau ga ada nanti fallback
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

def last_trading_close(df):
    df = df.sort_index()
    if "Volume" in df.columns:
        valid = df[df["Volume"] > 0]
        if not valid.empty:
            return float(get_col_series(valid, "Close").iloc[-1])
    return float(get_col_series(df, "Close").iloc[-1])

def normalize_voltype(v):
    """
    Bikin volume_type selalu jadi string sederhana:
    - kalau Series -> ambil value pertamanya
    - kalau ada newline -> ambil baris pertama aja
    """
    try:
        import pandas as pd
        if isinstance(v, pd.Series):
            # ambil value pertama yang bukan NaN
            v = v.dropna()
            if len(v) > 0:
                v = v.iloc[0]
            else:
                v = "normal"
    except Exception:
        pass

    v = str(v)
    if "\n" in v:
        v = v.splitlines()[0].strip()
    if v.lower() == "ticker":   # jaga-jaga kasus aneh
        v = "normal"
    return v

# =========================================================
# 1. INDIKATOR TEKNIKAL
# =========================================================

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

# =========================================================
# 1b. VOLUME CLASSIFIER (baru)
# =========================================================

def detect_volume_type(df: pd.DataFrame,
                       spike_ratio: float = 1.8) -> pd.DataFrame:
    """
    Kasih label volume per bar:
    - 'normal'
    - 'accumulation'  (spike + harga naik + close dekat high)
    - 'distribution'  (spike + harga turun + close dekat low)
    - 'churn'         (spike tapi candle kecil / tarik2an)
    Versi ini dibuat simpel supaya gak kena error 'Timestamp' object is not iterable.
    """
    df = df.copy()

    # pastikan kolom ini ada
    if "Volume" not in df.columns:
        df["volume_type"] = "normal"
        df["volume_tone"] = 0.0
        return df

    # MA volume 20 hari dari kolom Volume
    vol_ma_20 = df["Volume"].rolling(20).mean().fillna(0)

    # kekuatan candle (0 = close di low, 1 = close di high)
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    close_strength = ((df["Close"] - df["Low"]) / rng).clip(0, 1)

    # arah harga
    price_up = df["Close"] > df["Close"].shift(1)
    price_down = df["Close"] < df["Close"].shift(1)

    # volume spike?
    spike = df["Volume"] > (vol_ma_20 * spike_ratio)

    # vectorized label
    volume_type = np.where(
        spike & price_up & (close_strength >= 0.6),
        "accumulation",
        np.where(
            spike & price_down & (close_strength <= 0.4),
            "distribution",
            np.where(spike, "churn", "normal"),
        ),
    )

    df["volume_type"] = volume_type

    # encode ke angka biar aman buat model
    df["volume_tone"] = (
        (df["volume_type"] == "accumulation").astype(float) * 1.0
        + (df["volume_type"] == "churn").astype(float) * 0.5
        + (df["volume_type"] == "distribution").astype(float) * -1.0
    )

    return df

# =========================================================
# 2. DOWNLOAD & FEATURE ENGINEERING
# =========================================================

def get_daily(symbol="ADMR.JK", period="2y"):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return df
    df = df.tz_localize(None)
    return df

def make_features(df):
    df = df.copy()
    close = get_col_series(df, "Close")
    high  = get_col_series(df, "High")
    low   = get_col_series(df, "Low")
    vol   = get_col_series(df, "Volume")

    # ========= PRICE =========
    df["ret_1"]  = close.pct_change(1,  fill_method=None)
    df["ret_3"]  = close.pct_change(3,  fill_method=None)
    df["ret_5"]  = close.pct_change(5,  fill_method=None)
    df["ma_5"]   = close.rolling(5).mean()
    df["ma_20"]  = close.rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # ========= VOLUME =========
    df["vol_ma_5"]  = vol.rolling(5).mean()
    df["vol_ma_20"] = vol.rolling(20).mean()

    # hati-hati div 0
    vol_ma_10 = vol.rolling(10).mean()
    df["vol_ratio"] = vol / vol_ma_10.replace(0, np.nan)

    # indikator volume lain
    df["obv"] = compute_obv(close, vol)
    df["mfi"] = compute_mfi(high, low, close, vol, 14)
    df["adl"] = compute_adl(high, low, close, vol)

    # ========= VOLUME SPIKE (lama, tetap) =========
    df["vol_spike"] = vol / df["vol_ma_20"].replace(0, np.nan)
    df["is_spike"]  = (df["vol_spike"] > 3).astype(int)

    # ========= JENIS VOLUME (baru) =========
    # ini yang bikin kita bisa bedain ACC vs DIST vs CHURN
    df = detect_volume_type(df, spike_ratio=2.5)

    # ========= MOMENTUM / VOLATILITY =========
    df["rsi_14"] = compute_rsi(close, 14)
    macd, sig, hist = compute_macd(close)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd, sig, hist

    df["atr_14"] = compute_atr_like(high, low, close, 14)
    df["volatility_10"] = close.pct_change(fill_method=None).rolling(10).std()

    # ========= TARGET =========
    df["close_tomorrow"] = close.shift(-1)
    df["ret_tomorrow"]   = (df["close_tomorrow"] / close) - 1.0
    df["target_up"]      = (df["ret_tomorrow"] > 0.001).astype(int)

    # paling akhir
    return df.dropna()


# =========================================================
# 3. LIQUIDITY CHECK
# =========================================================

def assess_liquidity(df, days=20):
    """Balikin status likuiditas + avg volume + avg value (volume*close)."""
    if df.empty:
        return "N/A", 0.0, 0.0
    d = df.tail(days)
    avg_vol = float(d["Volume"].mean())
    avg_price = float(d["Close"].mean())
    avg_value = avg_vol * avg_price  # kira-kira IDR per hari

    # threshold bisa kamu sesuaikan
    if avg_value > 20_000_000_000:      # > 20 M / hari
        status = "LIQUID ✅"
    elif avg_value > 5_000_000_000:     # 5 - 20 M
        status = "MEDIUM ⚠️"
    else:
        status = "ILLQ ❌"
    return status, avg_vol, avg_value

# =========================================================
# 4. TRAIN MODEL XGBOOST
# =========================================================

def train_xgboost(df, show_metrics=True):
    # >>> new: masukin "volume_tone" ke fitur supaya model aware sama jenis volume
    feature_cols = [
        "ret_1", "ret_3", "ret_5",
        "ma_5", "ma_20", "ma_ratio",
        "vol_ratio", "vol_spike", "is_spike",
        "obv", "mfi", "adl",
        "rsi_14", "macd", "macd_sig", "macd_hist",
        "atr_14", "volatility_10",
        "volume_tone",          # <<< new
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

    if show_metrics:
        print("=== REGRESI RETURN (XGBoost) ===")
        print("MAE:", mean_absolute_error(y_ret_test, reg.predict(X_test)))
        print("R2 :", r2_score(y_ret_test, reg.predict(X_test)))
        print("\n=== KLASIFIKASI (Naik/Turun) ===")
        print(classification_report(y_cls_test, cls.predict(X_test), digits=4))

    return reg, cls, feature_cols

# =========================================================
# 5. KONFIRMASI & FINALIZER (1 SUMBER KEBENARAN)
# =========================================================

def confirm_signal_from_indicators(last_row, prob_up, ret_pred, volume_type=None):
    reasons = []
    score = 0
    max_score = 0

    # --- 0) hard cap AI rendah, tapi kita agak lunakkan ke 0.50
    hard_cap = None
    if prob_up < 0.50:
        hard_cap = "⚠️ LOW"
        reasons.append(f"AI prob rendah ({prob_up:.2%}) → cap ke LOW")

    # 1) AI prob
    max_score += 1
    if prob_up >= 0.65:
        score += 1
        reasons.append("AI prob tinggi (>= 65%)")
    elif prob_up >= 0.55:
        # tadinya langsung dianggap rendah, sekarang kita kasih poin setengah
        reasons.append("AI prob moderat (55–65%)")
    else:
        reasons.append("AI prob rendah (< 55%)")

    # 2) prediksi return
    max_score += 1
    if ret_pred > 0:
        score += 1
        reasons.append("Model prediksi naik")
    else:
        reasons.append("Model prediksi turun")

    # 3) RSI
    rsi_val = to_scalar(last_row["rsi_14"])
    max_score += 1
    rsi_ok = rsi_val >= 40
    if rsi_ok:
        score += 1
        reasons.append(f"RSI OK ({rsi_val:.1f} ≥ 40)")
    else:
        reasons.append(f"RSI lemah ({rsi_val:.1f} < 40)")

    # 4) MACD
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

    # 5) Volume ratio (turunin threshold ke 0.9 biar gak terlalu kejam)
    vol_ratio = to_scalar(last_row["vol_ratio"])
    max_score += 1
    vol_ok = vol_ratio >= 0.9
    if vol_ok:
        score += 1
        reasons.append(f"Volume OK (ratio {vol_ratio:.2f})")
    else:
        reasons.append(f"Volume agak rendah (ratio {vol_ratio:.2f})")

    # 6) Jenis volume – kasih boost kalau akumulasi
    if volume_type is not None:
        max_score += 1
        if volume_type == "accumulation":
            score += 1
            reasons.append("Volume spike → AKUMULASI ✅")
        elif volume_type == "distribution":
            reasons.append("Volume spike → DISTRIBUSI ⚠️")
        elif volume_type == "churn":
            reasons.append("Volume spike → CHURN (tarik2an)")
        else:
            reasons.append("Volume normal")

    confidence = (score / max_score) * 100

    if confidence >= 80:
        level = "✅ STRONG"
    elif confidence >= 60:
        level = "👍 OK"
    elif confidence >= 40:
        level = "⚠️ LOW"
    else:
        level = "❌ WEAK"

    # kalau dari awal AI rendah
    if hard_cap is not None:
        level = hard_cap
        confidence = min(confidence, 55)

    # kalau volume distribusi, jangan terlalu tinggi
    if volume_type == "distribution":
        level = "⚠️ LOW"
        confidence = min(confidence, 60)

    return {
        "score": score,
        "max_score": max_score,
        "confidence_pct": confidence,
        "level": level,
        "rsi_ok": rsi_ok,
        "macd_ok": macd_ok,
        "vol_ok": vol_ok,
        "vol_ratio": vol_ratio,
        "volume_type": volume_type,
        "reasons": reasons,
    }


def make_final_signal(prob_up, ret_pred, confirm, today_price, pred_price, atr):
    # campur AI dan indikator, jangan ambil yang paling rendah
    model_conf = prob_up
    confirm_conf = confirm["confidence_pct"] / 100

    base_conf = (model_conf + confirm_conf) / 2.0  # << tadinya min(...)

    # penalti
    penalty = 1.0
    if not confirm["vol_ok"]:
        penalty *= 0.9
    if not confirm["macd_ok"]:
        penalty *= 0.9
    if confirm.get("volume_type") == "distribution":
        penalty *= 0.8

    final_conf = round(base_conf * penalty, 2)

    # soft rule
    if model_conf >= 0.9 and final_conf < 0.55:
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


# =========================================================
# 6. PREDIKSI SATU SAHAM
# =========================================================

def predict_tomorrow(symbol, period="10y", silent=False):
    df = get_daily(symbol, period=period)
    if df.empty:
        if not silent:
            print(f"[{symbol}] ❌ Gagal download data.")
        return None

    # buang bar volume 0
    df = df[df["Volume"] > 0]

    feat = make_features(df)
    if feat.empty:
        if not silent:
            print(f"[{symbol}] ❌ Fitur kosong.")
        return None

    # train per saham
    reg, cls, feature_cols = train_xgboost(feat, show_metrics=not silent)

    last = feat.iloc[-1]
    X_live = last[feature_cols].values.reshape(1, -1)

    ret_pred = float(reg.predict(X_live)[0])
    prob_up = float(cls.predict_proba(X_live)[0][1])

    today = last_trading_close(df)
    pred_price = today * (1 + ret_pred)

    # liquidity info
    liquidity_status, avg_vol, avg_val = assess_liquidity(df)

    # jenis volume terakhir
    volume_type = str(last.get("volume_type", "normal"))

    # indikator
    confirm = confirm_signal_from_indicators(
        last, prob_up, ret_pred,
        volume_type=volume_type
    )
    atr_val = to_scalar(last["atr_14"])

    final_signal, final_conf, trade_plan = make_final_signal(
        prob_up=prob_up,
        ret_pred=ret_pred,
        confirm=confirm,
        today_price=today,
        pred_price=pred_price,
        atr=atr_val,
    )

    result = {
        "symbol": symbol,
        "signal": final_signal,
        "confidence": final_conf,
        "prob_up": prob_up,
        "pred_return": ret_pred,
        "today": today,
        "pred_price": pred_price,
        "volume_ratio": confirm["vol_ratio"],
        "volume_type": volume_type,   # <<< new
        "macd_ok": confirm["macd_ok"],
        "rsi_ok": confirm["rsi_ok"],
        "liquidity": liquidity_status,
        "avg_volume": avg_vol,
        "avg_value": avg_val,
        "trade_plan": trade_plan,
    }

    if not silent:
        print(f"\n=== {symbol} — FINAL AI SIGNAL ===")
        print(f"Signal         : {final_signal}")
        print(f"Confidence     : {final_conf:.2f}")
        print(f"Entry ideal    : {trade_plan['entry'][0]} – {trade_plan['entry'][1]}")
        print(f"Target jual    : {trade_plan['targets'][0]} – {trade_plan['targets'][1]}")
        print(f"Stop loss      : {trade_plan['stop_loss']}")
        print(f"Likuiditas     : {liquidity_status} (avg val ~ {avg_val:,.0f})")
        print("\n--- DETAIL PREDIKSI ---")
        print(f"Harga hari ini : {today:.2f}")
        print(f"Prediksi return: {ret_pred*100:.3f}%")
        print(f"Prediksi harga : {pred_price:.2f}")
        print(f"Prob. naik     : {prob_up:.2%}")
        print(f"Volume ratio   : {confirm['vol_ratio']:.2f}x")
        print(f"Jenis volume   : {volume_type}")
        print("\n--- AI CONFIRMATION ---")
        print(f"Level          : {confirm['level']} ({confirm['confidence_pct']:.1f}%)")
        for r in confirm["reasons"]:
            print(f"- {r}")

    return result

# =========================================================
# 7. SCREENER BANYAK SAHAM (REAL-TIME PRINT)
# =========================================================

def screen_stocks(tickers, period="10y"):
    results = []
    print("\n=== AI STOCK SCREENER (with liquidity & volume type) ===")
    for i, sym in enumerate(tickers, 1):
        try:
            res = predict_tomorrow(sym, period=period, silent=True)
            if res is None:
                print(f"[{i:3}] {sym:10} -> SKIP (no data)")
                continue

            results.append(res)

            # mapping volume type biar pendek
            vt = res.get("volume_type", "normal")
            if vt == "accumulation":
                vt_short = "ACC"
            elif vt == "distribution":
                vt_short = "DIST"
            elif vt == "churn":
                vt_short = "CHRN"
            else:
                vt_short = "-"

            # langsung print per saham
            print(
                f"[{i:3}] {res['symbol']:8} | {res['signal']:15} | "
                f"conf={res['confidence']:.2f} | prob={res['prob_up']:.2%} | "
                f"ret={res['pred_return']*100:5.2f}% | vol={res['volume_ratio']:.2f}x | "
                f"VTYPE={vt_short:4} | "
                f"{res['liquidity']:10} | MACD={'✅' if res['macd_ok'] else '-'} | "
                f"RSI={'✅' if res['rsi_ok'] else '-'}"
            )
        except Exception as e:
            print(f"[{i:3}] {sym:10} -> ERROR: {e}")
            continue

    # ranking
    signal_rank = {
        "BUY_STRONG": 4,
        "BUY_WAIT_VOLUME": 3,
        "WATCHLIST": 2,
        "NO_TRADE": 1,
    }
    results_sorted = sorted(
        results,
        key=lambda r: (
            signal_rank.get(r["signal"], 0),
            r["volume_ratio"],              # biar yang beneran rame ke atas
            r["prob_up"],
            r["confidence"],
        ),
        reverse=True,
    )

    # summary tabel
    print("\n=== SUMMARY TABLE ===")
    rows = []
    for i, r in enumerate(results_sorted, 1):
        vt_clean = normalize_voltype(r.get("volume_type", "-"))
        rows.append([
            i,
            r["symbol"],
            r["signal"],
            f"{r['confidence']:.2f}",
            f"{r['prob_up']:.2%}",
            f"{r['pred_return']*100:.2f}%",
            f"{r['volume_ratio']:.2f}x",
            vt_clean,
            r["liquidity"],
            "✅" if r["macd_ok"] else "-",
            "✅" if r["rsi_ok"] else "-",
        ])

    headers = ["No", "Symbol", "Signal", "Conf", "ProbUp", "Ret%", "VolRatio", "VolType", "Liquidity", "MACD", "RSI"]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(headers)
        for r in rows:
            print(r)

# =========================================================
# 8. MAIN
# =========================================================

if __name__ == "__main__":
    UNIVERSE = [
        # 🏦 BANKING & FINANCIAL
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BFIN.JK",
        "BTPS.JK", "AGRO.JK", "BBKP.JK", "BJBR.JK", "BJTM.JK", "PNBN.JK",
        "NISP.JK", "MEGA.JK", "ARTO.JK", "AMAR.JK", "BBYB.JK", "BCAP.JK",

        # ⚙️ MINING / COMMODITY / ENERGY
        "AMMN.JK", "ADMR.JK", "ADRO.JK", "PTBA.JK", "BYAN.JK", "HRUM.JK",
        "MDKA.JK", "MEDC.JK", "PGAS.JK", "ANTM.JK", "TINS.JK", "INCO.JK",
        "BREN.JK", "MBMA.JK", "BRMS.JK", "ELSA.JK", "ENRG.JK", "ITMG.JK",
        "ABMM.JK", "ESSA.JK", "CUAN.JK",

        # 🏗️ INFRASTRUCTURE / CONSTRUCTION / PROPERTY
        "WSKT.JK", "WIKA.JK", "PTPP.JK", "ADHI.JK", "SMGR.JK", "INTP.JK",
        "WTON.JK", "WIKA.JK", "JSMR.JK", "TOTL.JK", "PPRE.JK", "WSBP.JK",
        "CTRA.JK", "BSDE.JK", "PWON.JK", "ASRI.JK", "SMRA.JK", "LPKR.JK",
        "MDLN.JK", "KIJA.JK", "BEST.JK", "DILD.JK",

        # 📡 TELCO / TECHNOLOGY / DIGITAL
        "TLKM.JK", "ISAT.JK", "EXCL.JK", "MTEL.JK", "TOWR.JK",
        "BUKA.JK", "GOTO.JK", "DCII.JK", "DNET.JK", "EMTK.JK", "MCAS.JK",
        "ENVY.JK", "TFAS.JK", "HEAL.JK", "HATM.JK", "SATU.JK",

        # 🏭 INDUSTRIAL / MANUFACTURING
        "ASII.JK", "UNTR.JK", "IMAS.JK", "AUTO.JK", "SMSM.JK", "GJTL.JK",
        "INDS.JK", "INTA.JK", "KRAS.JK", "SCCO.JK", "KDSI.JK", "DPNS.JK",
        "BRPT.JK", "CHIP.JK", "POLY.JK", "MTDL.JK", "META.JK",

        # 🍜 CONSUMER GOODS / FMCG
        "UNVR.JK", "ICBP.JK", "INDF.JK", "MYOR.JK", "SIDO.JK",
        "HMSP.JK", "GGRM.JK", "DLTA.JK", "TCID.JK", "ROTI.JK",
        "CLEO.JK", "ULTJ.JK", "PRDA.JK", "KAEF.JK", "KLBF.JK",

        # 🛍️ RETAIL / TRADE / DISTRIBUTION
        "AMRT.JK", "ACES.JK", "MAPI.JK", "RALS.JK",
        "TELE.JK", "CSAP.JK", "DIVA.JK",

        # 🚢 LOGISTICS / TRANSPORTATION
        "ASSA.JK", "TMAS.JK", "SMDR.JK", "GIAA.JK", "IPCM.JK",
        "WINS.JK", "SAFE.JK", "HATM.JK",

        # 🧱 CEMENT / BUILDING MATERIALS
        "SMGR.JK", "INTP.JK", "SMCB.JK", "WTON.JK", "WSBP.JK",

        # 💊 HEALTHCARE / PHARMA
        "HEAL.JK", "MIKA.JK", "PRDA.JK", "SILO.JK", "KAEF.JK",
        "INAF.JK", "DVLA.JK", "TSPC.JK",

        # 💻 IT / SOFTWARE / SERVICES
        "MCAS.JK", "DIVA.JK", "MTDL.JK", "EDGE.JK", "PTSN.JK", "KIOS.JK",

        # 🧾 MISCELLANEOUS / CONGLOMERATES
        "LPKR.JK", "CTRA.JK", "EMTK.JK", "BIPI.JK", "MDIA.JK",
        "MNCN.JK", "SCMA.JK", "BMTR.JK", "VIVA.JK", "AKRA.JK"
    ]

    screen_stocks(UNIVERSE, period="10y")
