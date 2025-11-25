# =========================================================
# AI BREAKOUT PREDICTOR v2.1 — with Liquidity Filter
# By Bagus x Sophie 🧠🔥
# =========================================================
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
import warnings, os

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# KONFIGURASI
# =========================================================
LOG_FILE = "breakout_debug_log.txt"
OUT_FILE = "ai_breakout_results.csv"
MIN_ROWS = 120
PERIOD = "5y"
N_SPLITS = 5
MIN_LIQUIDITY = 3e10  # min avg traded value 30 miliar

# =========================================================
# UNIVERSE SAHAM
# =========================================================
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
        "WINS.JK", "SAFE.JK",

        # 🧱 CEMENT / BUILDING MATERIALS
        "SMGR.JK", "INTP.JK", "SMCB.JK", "WTON.JK", "WSBP.JK",

        # 💊 HEALTHCARE / PHARMA
        "HEAL.JK", "MIKA.JK", "PRDA.JK", "SILO.JK", "KAEF.JK",
        "INAF.JK", "DVLA.JK", "TSPC.JK",

        # 💻 IT / SOFTWARE / SERVICES
        "MCAS.JK", "DIVA.JK", "MTDL.JK", "EDGE.JK", "PTSN.JK", "KIOS.JK",

        # 🧾 MISCELLANEOUS / CONGLOMERATES
        "LPKR.JK", "CTRA.JK", "EMTK.JK", "BIPI.JK", "MDIA.JK",
        "MNCN.JK", "SCMA.JK", "BMTR.JK", "VIVA.JK", "AKRA.JK","BUVA.JK","PTRO.JK","IMPC.JK"
    ]

# =========================================================
# HELPER
# =========================================================
def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [c.title() for c in df.columns]
    return df

def download_data(symbol: str, period=PERIOD) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        if isinstance(df, tuple): df = df[0]
        if df.empty: return pd.DataFrame()
        df = _normalize_df(df)
        if "Volume" in df.columns: df = df[df["Volume"] > 0]
        df = df.tz_localize(None)
        return df
    except Exception as e:
        log(f"❌ Error {symbol}: {e}")
        return pd.DataFrame()

# =========================================================
# INDIKATOR TEKNIKAL
# =========================================================
def rsi(series, n=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, f=12, s=26, sig=9):
    ema1 = series.ewm(span=f, adjust=False).mean()
    ema2 = series.ewm(span=s, adjust=False).mean()
    m = ema1 - ema2
    signal = m.ewm(span=sig, adjust=False).mean()
    return m, signal, m - signal

def atr_like(h, l, c, n=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=14):
    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[df["Low"].diff() > 0] = 0
    tr = atr_like(df["High"], df["Low"], df["Close"], n)
    plus_di = 100 * (plus_dm.ewm(span=n).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(span=n).mean() / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.ewm(span=n).mean()

def obv(close, vol):
    return (np.sign(close.diff()) * vol).fillna(0).cumsum()

def bollinger_position(series, n=20):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = ma + 2*std
    lower = ma - 2*std
    return (series - lower) / (upper - lower)

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["ret_1"] = c.pct_change()
    df["ret_5"] = c.pct_change(5)
    df["rsi_14"] = rsi(c, 14)
    df["atr_14"] = atr_like(h, l, c, 14)
    df["adx_14"] = adx(df, 14)
    m, s, hst = macd(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = m, s, hst
    df["bb_pos"] = bollinger_position(c)
    df["obv"] = obv(c, v)
    df["vol_ratio"] = v / v.rolling(10).mean()
    df["vol_zscore"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    df["vol_breakout_ratio"] = v / v.rolling(20).max()

    # Liquidity metrics
    df["avg_value_20"] = (c * v).rolling(20).mean()
    df["liquidity_score"] = (df["avg_value_20"] - df["avg_value_20"].min()) / (
        df["avg_value_20"].max() - df["avg_value_20"].min()
    )

    # 🎯 Target: breakout besok
    df["target_breakout"] = (df["Close"].shift(-1) > df["High"].rolling(20).max()).astype(int)
    return df.dropna()

# =========================================================
# TRAINING
# =========================================================
def train_model(df_all: pd.DataFrame):
    features = [
        "ret_1","ret_5","rsi_14","atr_14","adx_14","macd","macd_sig","macd_hist",
        "bb_pos","obv","vol_ratio","vol_zscore","vol_breakout_ratio","liquidity_score"
    ]
    X, y = df_all[features], df_all["target_breakout"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=8,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    if len(y_test.unique()) > 1:
        print(classification_report(y_test, model.predict(X_test)))

    return model

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("🚀 Mulai AI Breakout Predictor v2.1 (Liquidity Aware)...")
    all_data = []

    for sym in UNIVERSE:
        df = download_data(sym)
        if df.empty:
            log(f"⚠️ {sym} kosong, skip.")
            continue
        feat = make_features(df)
        if feat.empty or len(feat) < MIN_ROWS:
            log(f"⚠️ {sym} fitur tidak cukup ({len(feat)}).")
            continue
        feat["symbol"] = sym
        all_data.append(feat)
        log(f"✅ {sym} OK ({len(feat)} baris)")

    if not all_data:
        log("❌ Tidak ada data valid.")
        exit()

    df_all = pd.concat(all_data, axis=0)
    model = train_model(df_all)

    # prediksi terakhir
    log("\n=== HASIL PREDIKSI BREAKOUT ===")
    results = []
    for sym in sorted(set(df_all["symbol"])):
        sub = df_all[df_all["symbol"] == sym]
        last = sub.iloc[-1]
        avg_value = last["avg_value_20"]

        if avg_value < MIN_LIQUIDITY:
            log(f"💧 {sym} skipped (illiquid, avg_value={avg_value:,.0f})")
            continue

        X_live = last[
            ["ret_1","ret_5","rsi_14","atr_14","adx_14","macd","macd_sig",
             "macd_hist","bb_pos","obv","vol_ratio","vol_zscore",
             "vol_breakout_ratio","liquidity_score"]
        ].values.reshape(1, -1)
        prob = float(model.predict_proba(X_live)[0][1])
        results.append({
            "Symbol": sym,
            "ProbBreakout": prob,
            "Close": last["Close"],
            "RSI": last["rsi_14"],
            "VolRatio": last["vol_ratio"],
            "BB_Pos": last["bb_pos"],
            "Liquidity(AvgRp)": avg_value
        })
        log(f"{sym:8} | ProbBreakout={prob:.2%} | Close={last['Close']:.2f} | AvgValue={avg_value/1e9:.1f}B")

    out = pd.DataFrame(results).sort_values("ProbBreakout", ascending=False)
    out.to_csv(OUT_FILE, index=False)
    log(f"\n📁 Hasil disimpan ke {OUT_FILE}")
    print(out.head(10))
