import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# CONFIG
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
        "WINS.JK", "SAFE.JK", "HATM.JK",

        # 🧱 CEMENT / BUILDING MATERIALS
        "SMGR.JK", "INTP.JK", "SMCB.JK", "WTON.JK", "WSBP.JK",

        # 💊 HEALTHCARE / PHARMA
        "HEAL.JK", "MIKA.JK", "SILO.JK", "KAEF.JK",
        "INAF.JK", "DVLA.JK", "TSPC.JK",

        # 💻 IT / SOFTWARE / SERVICES
        "MCAS.JK", "DIVA.JK", "MTDL.JK", "EDGE.JK", "PTSN.JK", "KIOS.JK",

        # 🧾 MISCELLANEOUS / CONGLOMERATES
        "LPKR.JK", "CTRA.JK", "EMTK.JK", "BIPI.JK", "MDIA.JK",
        "MNCN.JK", "SCMA.JK", "BMTR.JK", "VIVA.JK", "AKRA.JK"
    ]
VOLUME_SPIKE_FACTOR = 2.5
ROLLING_AVG_VOL = 10

# =========================================================
# UTILS
# =========================================================
def safe_series(x):
    """Paksa apa pun (Series/DataFrame/ndarray) jadi Series 1D."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] > 1:
            return x.iloc[:, 0]
        return x.squeeze()
    elif isinstance(x, pd.Series):
        return x
    else:
        return pd.Series(np.asarray(x).ravel())

# =========================================================
# INDICATORS
# =========================================================
def compute_rsi(close, period=14):
    close = safe_series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(period).mean()
    roll_down = loss.rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def bollinger_bands(close, period=20, num_std=2):
    close = safe_series(close).astype(float)
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return safe_series(upper), safe_series(lower)

# =========================================================
# LOAD
# =========================================================
def load_symbol_df(symbol, period="1y"):
    data = yf.download(symbol, period=period, interval="1d", progress=False, group_by='ticker')
    if data is None or data.empty:
        return pd.DataFrame()

    # kalau hasil multi-index (Close, SYMBOL) → ratakan jadi 1 kolom
    if isinstance(data.columns, pd.MultiIndex):
        df = data.xs(symbol, axis=1, level=0)
    else:
        df = data

    df = df.sort_index()
    df["Symbol"] = symbol
    df["Close"] = safe_series(df["Close"])
    df["Volume"] = safe_series(df["Volume"])

    df["Return"] = df["Close"].pct_change()
    df["VolRatio5"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["VolRatio10"] = df["Volume"] / df["Volume"].rolling(10).mean()
    df["RSI14"] = compute_rsi(df["Close"], 14)

    bb_u, bb_l = bollinger_bands(df["Close"])
    df["BB_upper"] = safe_series(bb_u)
    df["BB_lower"] = safe_series(bb_l)
    denom = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_pos"] = (df["Close"] - df["BB_lower"]) / denom
    df["BB_pos"].replace([np.inf, -np.inf], np.nan, inplace=True)

    df["VolAvg10"] = df["Volume"].rolling(ROLLING_AVG_VOL).mean()
    df["TargetSpike"] = ((df["Volume"].shift(-1) / df["VolAvg10"]) > VOLUME_SPIKE_FACTOR).astype(int)
    df.dropna(inplace=True)
    return df

# =========================================================
# BUILD
# =========================================================
def build_global_dataset(symbols):
    frames = []
    for sym in symbols:
        try:
            df = load_symbol_df(sym)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"[WARN] gagal load {sym}: {e}")
    return pd.concat(frames, axis=0).sort_index() if frames else pd.DataFrame()

# =========================================================
# TRAIN
# =========================================================
def train_model(df):
    feature_cols = ["Return","VolRatio5","VolRatio10","RSI14","BB_pos"]
    df = df.dropna(subset=feature_cols + ["TargetSpike"])
    if df.empty:
        raise ValueError("Dataset kosong setelah dropna.")

    df = df.sort_index()
    split_idx = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train_df[feature_cols], train_df["TargetSpike"]
    X_test, y_test = test_df[feature_cols], test_df["TargetSpike"]

    model = XGBClassifier(
        max_depth=4, learning_rate=0.12, n_estimators=280,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss", n_jobs=4
    )
    model.fit(X_train, y_train)
    from sklearn.metrics import classification_report
    y_pred = (model.predict_proba(X_test)[:,1] > 0.5).astype(int)
    print("=== MODEL EVALUATION ===")
    print(classification_report(y_test, y_pred, digits=4))
    return model, feature_cols

# =========================================================
# PREDICT
# =========================================================
def predict_tomorrow_spike(model, feature_cols, symbols):
    results = []
    for sym in symbols:
        try:
            df = load_symbol_df(sym, period="6mo")
            if df.empty: continue
            last = df.iloc[-1]
            x = last[feature_cols].values.reshape(1, -1)
            prob = model.predict_proba(x)[0][1]
            results.append({
                "Symbol": sym,
                "Date": last.name.strftime("%Y-%m-%d"),
                "Close": float(last["Close"]),
                "VolToday": int(last["Volume"]),
                "ProbSpikeTomorrow": prob
            })
        except Exception as e:
            print(f"[WARN] gagal prediksi {sym}: {e}")
    return sorted(results, key=lambda r: r["ProbSpikeTomorrow"], reverse=True)

# =========================================================
# MAIN
# =========================================================
def main():
    print("Mengambil data & membangun dataset...")
    df = build_global_dataset(UNIVERSE)
    if df.empty:
        print("Dataset kosong!")
        return

    print(f"Total baris: {len(df)}")
    model, feats = train_model(df)
    results = predict_tomorrow_spike(model, feats, UNIVERSE)

    if HAS_TABULATE:
        print(tabulate([
            [r["Symbol"], r["Date"], f"{r['Close']:.2f}", f"{r['VolToday']:,}", f"{r['ProbSpikeTomorrow']*100:5.2f} %"]
            for r in results
        ], headers=["Symbol","Date","Close","Volume Today","Prob. Spike Tomorrow"], tablefmt="github"))
    else:
        for r in results:
            print(f"{r['Symbol']:8s} | {r['Date']} | Close {r['Close']:.2f} | Vol {r['VolToday']:,} | Prob Spike: {r['ProbSpikeTomorrow']*100:5.2f}%")

    print("\nTop 5 kandidat spike besok:")
    for r in results[:5]:
        print(f"- {r['Symbol']} → {r['ProbSpikeTomorrow']*100:5.2f}%")

if __name__ == "__main__":
    main()
