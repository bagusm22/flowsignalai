# screener_gapup_live.py
# ------------------------------------------------------------
# AI Screener: Buy Sore (15:30-15:59) -> Sell Pagi (09:00)
# Fokus: prediksi GAP UP besok pagi (Open H+1 > Close H)
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier, XGBRegressor  # regressor optional (jika mau prediksi besaran gap)
from datetime import datetime, time
import pytz

# optional
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# =========================
# 0) UTIL & CONSTANTS
# =========================

IDX_TZ = "Asia/Jakarta"

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
    """pastikan volume_type jadi string 1 baris (bukan Series / multi-line)."""
    try:
        if isinstance(v, pd.Series):
            v = v.dropna()
            v = v.iloc[0] if len(v) else "normal"
    except Exception:
        pass
    v = str(v)
    if "\n" in v:
        v = v.splitlines()[0].strip()
    if v.lower() == "ticker":
        v = "normal"
    return v

# =========================
# 1) MARKET PROGRESS (IDX)
# =========================

def get_idx_session_progress():
    """
    Estimasi progress market IDX (0.0 - 1.0)
    - Sesi 1: 09:00–11:30 (150 menit)
    - Sesi 2: 14:00–15:59 (119 menit, kita bulatkan 120)
    Total ~ 270 menit (kita pakai 270 biar simple)
    """
    tz = pytz.timezone(IDX_TZ)
    now = datetime.now(tz).time()

    s1_start = time(9, 0)
    s1_end   = time(11, 30)
    s2_start = time(14, 0)
    s2_end   = time(15, 59)  # mendekati penutupan

    total_minutes = 270

    def minutes_between(t1, t2):
        return (t2.hour - t1.hour) * 60 + (t2.minute - t1.minute)

    if now < s1_start:
        return 0.0
    if s1_start <= now <= s1_end:
        done = minutes_between(s1_start, now)
        return done / total_minutes
    if s1_end < now < s2_start:
        done = minutes_between(s1_start, s1_end)
        return done / total_minutes
    if s2_start <= now <= s2_end:
        done = minutes_between(s1_start, s1_end) + minutes_between(s2_start, now)
        return done / total_minutes
    return 1.0

def scale_last_volume_for_intraday(df, min_prog=0.25):
    """
    Skala volume bar terakhir saat market belum tutup.
    Tujuan: supaya perbandingan vs MA20 tidak 'kecil' terus di siang/sore.
    """
    prog = get_idx_session_progress()
    if prog <= 0.0 or prog >= 1.0:
        return df

    df = df.copy()
    safe_prog = max(prog, min_prog)
    scale_factor = 1.0 / safe_prog
    last_idx = df.index[-1]
    df.loc[last_idx, "Volume"] = df.loc[last_idx, "Volume"] * scale_factor
    return df

# =========================
# 2) INDIKATOR TEKNIKAL
# =========================

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

# =========================
# 3) VOLUME CLASSIFIER
# =========================

def detect_volume_type(df: pd.DataFrame, spike_ratio: float = 1.6) -> pd.DataFrame:
    """
    Klasifikasi volume per bar:
    - accumulation: spike + up + close dekat high
    - distribution: spike + down + close dekat low
    - churn: spike tapi arah ambigu / tarik2an
    - normal: selain itu
    spike_ratio lebih agresif (1.6) untuk sesi live sore.
    """
    df = df.copy()
    if "Volume" not in df.columns:
        df["volume_type"] = "normal"
        df["volume_tone"] = 0.0
        return df

    vol_ma_20 = df["Volume"].rolling(20).mean().fillna(0)

    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    close_strength = ((df["Close"] - df["Low"]) / rng).clip(0, 1)

    price_up = df["Close"] > df["Close"].shift(1)
    price_down = df["Close"] < df["Close"].shift(1)

    spike = df["Volume"] > (vol_ma_20 * spike_ratio)

    volume_type = np.where(
        spike & price_up & (close_strength >= 0.55), "accumulation",
        np.where(
            spike & price_down & (close_strength <= 0.45), "distribution",
            np.where(spike, "churn", "normal"),
        ),
    )

    df["volume_type"] = volume_type
    df["volume_tone"] = (
        (df["volume_type"] == "accumulation").astype(float) * 1.0
        + (df["volume_type"] == "churn").astype(float) * 0.5
        + (df["volume_type"] == "distribution").astype(float) * -1.0
    )
    return df

# =========================
# 4) DATA & FEATURES
# =========================

def get_daily(symbol="BBCA.JK", period="5y"):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return df
    return df.tz_localize(None)

def make_features(df):
    df = df.copy()
    close = get_col_series(df, "Close")
    high  = get_col_series(df, "High")
    low   = get_col_series(df, "Low")
    vol   = get_col_series(df, "Volume")

    # PRICE
    df["ret_1"] = close.pct_change(1, fill_method=None)
    df["ret_3"] = close.pct_change(3, fill_method=None)
    df["ret_5"] = close.pct_change(5, fill_method=None)
    df["ma_5"]  = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # VOLUME
    df["vol_ma_5"]  = vol.rolling(5).mean()
    df["vol_ma_20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / vol.rolling(10).mean().replace(0, np.nan)

    df["obv"] = compute_obv(close, vol)
    df["mfi"] = compute_mfi(high, low, close, vol, 14)
    df["adl"] = compute_adl(high, low, close, vol)

    # Volume spike (lama)
    df["vol_spike"] = vol / df["vol_ma_20"].replace(0, np.nan)
    df["is_spike"]  = (df["vol_spike"] > 3).astype(int)

    # Jenis volume (baru)
    df = detect_volume_type(df, spike_ratio=1.6)

    # MOMENTUM & VOLATILITY
    df["rsi_14"] = compute_rsi(close, 14)
    macd, sig, hist = compute_macd(close)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd, sig, hist

    df["atr_14"] = compute_atr_like(high, low, close, 14)
    df["volatility_10"] = close.pct_change(fill_method=None).rolling(10).std()

    # TARGETS utk gap-up (Open besok vs Close hari ini)
    df["open_tomorrow"] = get_col_series(df, "Open").shift(-1)
    df["gap_return"]    = (df["open_tomorrow"] / close) - 1.0
    df["target_gap_up"] = (df["gap_return"] > 0.005).astype(int)  # contoh threshold 0.5%

    return df.dropna()

# =========================
# 5) LIQUIDITY
# =========================

def assess_liquidity(df, days=20):
    if df.empty:
        return "N/A", 0.0, 0.0
    d = df.tail(days)
    avg_vol = float(d["Volume"].mean())
    avg_price = float(d["Close"].mean())
    avg_value = avg_vol * avg_price
    if avg_value > 20_000_000_000:
        status = "LIQUID ✅"
    elif avg_value > 5_000_000_000:
        status = "MEDIUM ⚠️"
    else:
        status = "ILLQ ❌"
    return status, avg_vol, avg_value

# =========================
# 6) MODEL GAP-UP (CLASSIFIER)
# =========================

GAP_FEATURES = [
    "ret_1","ret_3","ret_5",
    "ma_5","ma_20","ma_ratio",
    "vol_ratio","vol_spike","is_spike",
    "obv","mfi","adl",
    "rsi_14","macd","macd_sig","macd_hist",
    "atr_14","volatility_10",
    "volume_tone",
]

def train_gapup_model(df, show_metrics=False):
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

    X = df[GAP_FEATURES]
    y = df["target_gap_up"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = XGBClassifier(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)

    if show_metrics and len(X_test) > 0:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        try:
            print("Accuracy :", accuracy_score(y_test, y_pred))
            print("ROC AUC  :", roc_auc_score(y_test, y_proba))
        except Exception:
            pass
        print(classification_report(y_test, y_pred, digits=4))

    return clf

# =========================
# 7) CONFIRMATION & FINAL
# =========================

def confirm_signal(last_row, prob_up):
    reasons = []
    score = 0
    max_score = 0

    # AI Prob
    max_score += 1
    if prob_up >= 0.7:
        score += 1
        reasons.append("Prob gap-up tinggi (>= 70%)")
    else:
        reasons.append("Prob gap-up moderat")

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

    # Volume tone
    vt = normalize_voltype(last_row.get("volume_type", "normal"))
    max_score += 1
    if vt == "accumulation":
        score += 1
        reasons.append("Volume spike → AKUMULASI ✅")
    elif vt == "distribution":
        reasons.append("Volume spike → DISTRIBUSI ⚠️")
    elif vt == "churn":
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

    return {
        "level": level,
        "confidence_pct": confidence,
        "rsi_ok": rsi_ok,
        "macd_ok": macd_ok,
        "vol_type": vt,
        "reasons": reasons,
    }

# =========================
# 8) PREDIKSI SATU SAHAM (LIVE 15:30+)
# =========================

def predict_gap_up_live(symbol, period="5y", show_metrics=False):
    df = get_daily(symbol, period=period)
    if df.empty:
        return None

    # filter bar volume 0 (jika ada anomali)
    df = df[df["Volume"] > 0]

    # scale volume bar terakhir agar setara 'end of day'
    df = scale_last_volume_for_intraday(df)

    feat = make_features(df)
    if feat.empty or len(feat) < 50:
        return None

    # train pakai data sampai kemarin aja
    feat_train = feat.iloc[:-1].copy()
    live_row   = feat.iloc[-1].copy()

    model = train_gapup_model(feat_train, show_metrics=show_metrics)

    X_live = live_row[GAP_FEATURES].values.reshape(1, -1)
    prob_up = float(model.predict_proba(X_live)[0][1])

    today_close = last_trading_close(df)
    liquidity_status, avg_vol, avg_val = assess_liquidity(df)

    confirm = confirm_signal(live_row, prob_up=prob_up)

    return {
        "symbol": symbol,
        "prob_gap_up": prob_up,
        "confirm_level": confirm["level"],
        "confidence": round(confirm["confidence_pct"]/100, 2),
        "macd_ok": confirm["macd_ok"],
        "rsi_ok": confirm["rsi_ok"],
        "volume_type": confirm["vol_type"],
        "today_close": today_close,
        "liquidity": liquidity_status,
        "avg_value": avg_val,
    }

# =========================
# 9) SCREENER
# =========================

def screen_universe(tickers, period="5y", prob_threshold=0.7):
    results = []
    print("\n=== GAP-UP SCREENER (LIVE 15:30–CLOSE) ===")
    for i, sym in enumerate(tickers, 1):
        try:
            res = predict_gap_up_live(sym, period=period, show_metrics=False)
            if res is None:
                print(f"[{i:3}] {sym:8} -> SKIP (no data)")
                continue

            vt = normalize_voltype(res.get("volume_type", "normal"))
            vt_short = {"accumulation":"ACC","distribution":"DIST","churn":"CHRN"}.get(vt, "-")

            # Print per baris (ringkas)
            print(
                f"[{i:3}] {res['symbol']:8} | prob={res['prob_gap_up']:.2%} | "
                f"CONF={res['confidence']:.2f} ({res['confirm_level']}) | "
                f"VTYPE={vt_short:4} | "
                f"MACD={'✅' if res['macd_ok'] else '-'} | RSI={'✅' if res['rsi_ok'] else '-'} | "
                f"{res['liquidity']}"
            )

            # simpan hanya kandidat di atas threshold
            if res["prob_gap_up"] >= prob_threshold:
                results.append(res)

        except Exception as e:
            print(f"[{i:3}] {sym:8} -> ERROR: {e}")

    # sort hasil
    results_sorted = sorted(
        results,
        key=lambda r: (r["prob_gap_up"], r["confidence"], r["avg_value"]),
        reverse=True
    )

    # Summary table
    print("\n=== SUMMARY (KANDIDAT BUY SORE → SELL PAGI) ===")
    rows = []
    for i, r in enumerate(results_sorted, 1):
        rows.append([
            i,
            r["symbol"],
            f"{r['prob_gap_up']:.2%}",
            f"{r['confidence']:.2f}",
            r["confirm_level"],
            r["volume_type"],
            "✅" if r["macd_ok"] else "-",
            "✅" if r["rsi_ok"] else "-",
            r["liquidity"],
            f"{r['avg_value']:,.0f}",
        ])

    headers = ["No","Symbol","ProbGapUp","Conf","Level","VolType","MACD","RSI","Liquidity","AvgVal(IDR)"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(headers)
        for row in rows:
            print(row)

    return results_sorted

# =========================
# 10) MAIN
# =========================

if __name__ == "__main__":
    # Universe ringkas (boleh tambah/kurangi)
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
        "TELE.JK", "CSAP.JK", "DIVA.JK", "MAPA.JK",

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
        "LPKR.JK", "EMTK.JK", "BIPI.JK", "MDIA.JK",
        "MNCN.JK", "SCMA.JK", "BMTR.JK", "VIVA.JK", "AKRA.JK"
    ]

    # Threshold kandidat
    PROB_THRESHOLD = 0.70  # 70%+

    # Jalankan screener (idealnya jam 15:30–15:59 WIB)
    screen_universe(UNIVERSE, period="5y", prob_threshold=PROB_THRESHOLD)
