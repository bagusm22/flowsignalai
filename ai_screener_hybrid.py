import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings, os, traceback

warnings.filterwarnings("ignore", category=FutureWarning)

LOG_FILE = "debug_log.txt"
MIN_ROWS = 120

FEATURE_COLS = [
    "ret_1","ret_3","ret_5",
    "ma_ratio",
    "vol_ratio",
    "volatility_10",
    "atr_14",
    "rsi_14",
    "macd","macd_sig","macd_hist",
    "is_monday",
    "mkt_ret_1","rel_mkt_1",
    "sec_ret_1","rel_sec_1",
]

def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


# =========================================================
# 1. DOWNLOAD DENGAN FIX MULTIINDEX
# =========================================================
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


def download_market(period="3y"):
    return download_single("^JKSE", period)


# =========================================================
# 2. INDIKATOR
# =========================================================
def rsi(series, n=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series, f=12, s=26, sig=9):
    ema1 = series.ewm(span=f, adjust=False).mean()
    ema2 = series.ewm(span=s, adjust=False).mean()
    m = ema1 - ema2
    signal = m.ewm(span=sig, adjust=False).mean()
    return m, signal, m - signal


def atr_like(h, l, c, n=14):
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean().fillna(0)


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================
def make_features(df: pd.DataFrame, mkt: pd.DataFrame = None) -> pd.DataFrame:
    needed = {"High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        # kalau sampai sini masih kurang kolom, kita stop aja
        missing = needed - set(df.columns)
        log(f"⚠️ kolom wajib hilang: {missing}")
        return pd.DataFrame()

    close = df["Close"]
    high = df["high"] if "high" in df.columns else df["High"]
    low = df["low"] if "low" in df.columns else df["Low"]
    vol = df["Volume"]

    # price/momentum
    df["ret_1"] = close.pct_change()
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # volume
    df["vol_ratio"] = vol / vol.rolling(10).mean()
    df["volatility_10"] = close.pct_change().rolling(10).std()

    # volat & osc
    df["atr_14"] = atr_like(high, low, close, 14)
    df["rsi_14"] = rsi(close, 14)
    m_val, s_val, h_val = macd(close)
    df["macd"], df["macd_sig"], df["macd_hist"] = m_val, s_val, h_val

    # calendar
    df["is_monday"] = (df.index.weekday == 0).astype(int)

    # market context
    if mkt is not None and not mkt.empty and "Close" in mkt.columns:
        mkt = mkt.reindex(df.index)
        mret = mkt["Close"].pct_change()
        df["mkt_ret_1"] = mret
        df["rel_mkt_1"] = df["ret_1"] - mret
    else:
        df["mkt_ret_1"] = 0.0
        df["rel_mkt_1"] = 0.0

    # untuk sekarang sektor nol-in aja
    df["sec_ret_1"] = 0.0
    df["rel_sec_1"] = 0.0

    # target
    df["ret_tomorrow"] = close.shift(-1) / close - 1
    df["rank_next"] = df["ret_tomorrow"].rank(pct=True)
    df["target_top"] = (df["rank_next"] >= 0.9).astype(int)

    # jangan drop semua NaN sekaligus, pilih subset
    use_cols = [c for c in FEATURE_COLS if c in df.columns] + ["target_top", "ret_tomorrow"]
    df = df.dropna(subset=use_cols, how="any")

    return df


# =========================================================
# 4. TRAINING
# =========================================================
def train_model(all_df: pd.DataFrame):
    if all_df.empty:
        return None

    X = all_df[FEATURE_COLS]
    y = all_df["target_top"]

    split = int(len(X) * 0.8)
    Xtr, Xts = X.iloc[:split], X.iloc[split:]
    ytr, yts = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(Xtr, ytr)

    if len(yts) > 0 and yts.nunique() > 1:
        rep = classification_report(yts, model.predict(Xts), digits=4)
        log("=== MODEL REPORT ===")
        log(rep)
    else:
        log("⚠️ test set kosong / cuma 1 kelas, skip report.")

    return model


# =========================================================
# 5. SCREENER
# =========================================================
def screen_stocks(universe, period="10y"):
    # bersihin log lama
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("⏬ Download market...")
    mkt = download_market(period)

    log("⏬ Download saham...")
    feats = []

    for sym in universe:
        df = download_single(sym, period)
        if df.empty:
            log(f"⚠️ {sym} kosong / gagal download.")
            continue

        feat = make_features(df, mkt)
        if feat.empty:
            log(f"⚠️ {sym} fitur gagal dibuat.")
            continue
        if len(feat) < MIN_ROWS:
            log(f"⚠️ {sym} data terlalu sedikit ({len(feat)}).")
            continue

        feat = feat.copy()  # ini penting banget, memastikan dia DataFrame baru
        feat.loc[:, "symbol"] = sym
        feats.append(feat)
        log(f"✅ {sym} fitur OK ({len(feat)} baris)")

    if not feats:
        log("❌ Tidak ada saham valid.")
        return

    all_df = pd.concat(feats, axis=0)
    model = train_model(all_df)
    if model is None:
        log("❌ gagal train model.")
        return

    log("\n=== HASIL PREDIKSI ===")
    results = []
    for sym in sorted(set(all_df["symbol"])):
        sub = all_df[all_df["symbol"] == sym]
        last = sub.iloc[-1]
        X_live = last[FEATURE_COLS].values.reshape(1, -1)
        prob = float(model.predict_proba(X_live)[0][1])
        pred_ret = float(last["ret_tomorrow"])
        results.append((sym, prob, pred_ret))
        log(f"{sym:8} | ProbTop={prob:.2%} | RetPred={pred_ret*100:.2f}%")

    # simpan ke csv
    out = pd.DataFrame(results, columns=["Symbol", "ProbTop", "PredictedRet"]).sort_values("ProbTop", ascending=False)
    out.to_csv("ai_screener_results.csv", index=False)
    log("\n📁 Hasil disimpan ke ai_screener_results.csv")


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
    screen_stocks(UNIVERSE)
