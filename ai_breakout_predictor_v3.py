#!/usr/bin/env python3
"""
Hybrid Breakout Bullish Prediction
(daily + 1H intraday confirm, SAFE columns, SAFE timezone)
"""
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================

UNIVERSE = ["AADI","AALI","ABBA","ABDA","ABMM","ACES","ACRO","ACST","ADCP","ADES","ADHI","ADMF","ADMG","ADMR","ADRO","AEGS","AGAR","AGII","AGRO","AGRS","AHAP","AIMS","AISA","AKKU","AKPI","AKRA","AKSI","ALDO","ALII","ALKA","AMAG","AMAN","AMAR","AMFG","AMIN","AMMN","AMOR","AMRT","ANDI","ANJT","ANTM","APEX","APIC","APLI","ARGO","ARII","ARKA","ARMY","ARNA","ARTA","ARTO","ASBI","ASDM","ASGR","ASHA","ASII","ASJT","ASMI","ASPI","ASRM","ASSA","ATAP","ATIC","AUTO","AVIA","AXIO","BACA","BAJA","BALI","BANK","BAPA","BAPI","BATA","BAYU","BBCA","BBHI","BBKP","BBLD","BBMD","BBNI","BBRI","BBRM","BBSI","BBSS","BBTN","BBYB","BCAP","BCIC","BCIP","BDMN","BEEF","BEKS","BELL","BEST","BFIN","BGTG","BHAT","BHIT","BIMA","BINA","BIPI","BIRD","BISI","BJBR","BJTM","BKDP","BKSL","BKSW","BLTA","BLTZ","BLUE","BMAS","BMHS","BMRI","BMSR","BMTR","BOGA","BOGA","BOLA","BOLT","BOSS","BPFI","BPII","BPTR","BRAM","BRIS","BRMS","BRNA","BRPT","BSDE","BSIM","BSSR","BTEK","BTEL","BTON","BTPN","BTPS","BUDI","BUKA","BUKK","BULL","BUVA","BVIC","BWPT","BYAN","CAKK","CAMP","CANI","CARE","CARS","CASA","CASH","CEKA","CENT","CFIN","CINT","CITY","CLEO","CLPI","CMNP","CMPP","CMRY","CMNT","COCO","COAL","COWL","CPIN","CPRI","CPRO","CSAP","CSIS","CSMI","CSRA","CTBN","CTRA","CTTH","CUAN","CYBR","DADA","DART","DAYA","DCII","DEAL","DEFI","DEPO","DEWA","DFAM","DGIK","DIGI","DILD","DIVA","DKFT","DLTA","DMAS","DMND","DNAR","DNET","DOID","DPNS","DPUM","DRMA","DSFI","DSNG","DSSA","DUCK","DUCK","DUTI","DVLA","DWGL","DYAN","EAST","ECII","EDGE","EKAD","ELIT","ELSA","ELTY","EMDE","EMTK","ENRG","ENVY","EPAC","EPMT","ERAA","ETWA","EXCL","FAPA","FAST","FISH","FITT","FMII","FOOD","FORU","FPNI","GAMA","GDST","GDYR","GEMA","GEMS","GGRM","GGRP","GHON","GJTL","GLOB","GMFI","GMTD","GPRA","GSMF","GTBO","GULA","GZCO","HADE","HAIS","HAIS","HATM","HDFA","HEAL","HELI","HERO","HEXA","HKMU","HMSP","HOKI","HOME","HOTL","HRME","HRTA","HRUM","IATA","IBFN","IBOS","IBST","ICBP","ICON","IDEA","IDPR","IFII","IGAR","IIKP","IKAI","IKBI","IMAS","IMJS","IMPC","INAF","INAI","INCF","INCI","INCO","INDF","INDR","INDS","INDX","INDY","INKP","INOV","INPC","INPP","INPS","INRU","INTA","INTD","INTP","IPAC","IPCC","IPOL","IPPE","IRRA","ISAT","ISSP","ITIC","ITMG","JAST","JAWA","JAYA","JECC","JGLE","JIHD","JKON","JKSW","JLBI","JMTO","JPFA","JRPT","JSKY","JSMR","JSPT","JTPE","KAEF","KARW","KAYU","KBAG","KBLI","KBLM","KBLV","KBRI","KDSI","KEEN","KEJU","KIAS","KICI","KIJA","KINO","KIOS","KJEN","KKGI","KLBF","KMDS","KMTR","KOBX","KOIN","KONI","KOPI","KOTA","KPAL","KPAS","KPIG","KPPI","KRAH","KRAH","KRAS","KREN","KRYA","KUAS","LABA","LAND","LAPD","LATX","LCGP","LCKM","LEAD","LIFE","LINK","LION","LIPP","LMAS","LMPI","LMWG","LPCK","LPGI","LPIN","LPKR","LPLI","LPPF","LPPS","LRNA","LSIP","LTLS","LUCK","LUXI","MABA","MAGP","MAHA","MAIN","MAMI","MAPA","MAPB","MAPI","MARI","MARK","MASA","MASB","MAYA","MBAP","MBMA","MBSS","MBTO","MCAS","MCOR","MDIA","MDKA","MDKI","MDLN","MDTL","MEDC","MEGA","MERK","META","MFIN","MFMI","MGNA","MGRO","MINA","MITI","MKPI","MKUA","MLBI","MLIA","MLPL","MLPT","MMLP","MNCN","MOLI","MPMX","MPPA","MPRO","MRAT","MREI","MSIN","MSKY","MTDL","MTFN","MTLA","MTPS","MTRA","MTWI","MYOH","MYOR","MYRX","NASA","NASI","NATO","NBLS","NELY","NFCX","NGAP","NICL","NICK","NIKL","NISP","NOBU","NRCA","NUSA","NZIA","OASA","OCAP","OKAS","OMRE","OPMS","PADI","PALM","PAMG","PANR","PANS","PBID","PBRX","PCAR","PDES","PEGE","PEHA","PGAS","PGJO","PGLI","PGUN","PICO","PJAA","PKPK","PLAN","PLAS","PLIN","PMJS","PMMP","PNBN","PNBS","PNIN","PNLF","PNSE","POLA","POLI","POLU","POLY","POOL","PORT","POWR","PPGL","PPIN","PPRE","PPRO","PPSD","PPTP","PRAS","PRDA","PRIM","PSAB","PSDN","PSGO","PSKT","PSSI","PTBA","PTIS","PTPP","PTRO","PTSN","PTSP","PUDP","PURE","PURI","PWON","PYFA","PZZA","RAJA","RALS","RANC","RBMS","RDTX","REAL","RELI","RICY","RIGS","RIMO","RISE","RMBA","RODA","ROTI","RUIS","SAFE","SAME","SAPX","SATU","SCCO","SCMA","SCNP","SDMU","SDPC","SDRA","SDSA","SEMA","SENT","SHID","SHIP","SIDO","SILO","SIMA","SIMP","SINI","SIPD","SIPD","SIRE","SKBM","SKLT","SKRN","SKYB","SLIS","SMAR","SMBR","SMCB","SMDM","SMDR","SMGR","SMKL","SMMA","SMMT","SMRA","SMRU","SMTM","SOBI","SOFA","SONA","SOSS","SOTS","SPMA","SPTO","SQMI","SRAJ","SRIL","SRSN","SSIA","SSMS","SSTM","STAR","STTP","SUGI","SULI","SUPR","SURE","SWAT","TALF","TAMA","TAMU","TAPG","TBIG","TBLA","TBMS","TCID","TCPI","TDPM","TEBE","TELE","TFCO","TGKA","TGRA","TIFA","TINS","TIRA","TIRT","TKIM","TLKM","TMAS","TMPI","TMPO","TMSR","TNCA","TOBA","TOPS","TOTL","TOTO","TOWR","TPIA","TPMA","TRAM","TRIL","TRIM","TRIO","TRIS","TRJA","TRUK","TRUS","TUGU","TURI","UANG","UCID","ULTJ","UNIC","UNIT","UNSP","UNTR","UNVR","URBN","VICI","VINS","VIVA","VOKS","VRNA","VTNY","WEGE","WEHA","WICO","WIKA","WINS","WIRG","WOMF","WOOD","WSBP","WSKT","WTON","YELO","YPAS","YULE","ZBRA","ZINC","ZONE"]
# UNIVERSE = ["BBCA"]
UNIVERSE = [t + ".JK" for t in UNIVERSE]

PERIOD_DAILY = "3y"
PERIOD_INTRADAY = "30d"
INTERVAL_INTRADAY = "60m"
LIQUIDITY = 2e10


# =========================
# HELPERS
# =========================
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def get_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col


# =========================
# DOWNLOADERS
# =========================
def download_daily(symbol: str, period: str = PERIOD_DAILY):
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        # print(df)
        if df is None or df.empty:
            return None
        df = normalize_df(df)
        
        # daily biasanya sudah tz-naive, kita pastikan
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.dropna(how="any")
    except Exception:
        # diem aja, jangan print apapun
        return None


def download_intraday(symbol: str, period: str = PERIOD_INTRADAY, interval: str = INTERVAL_INTRADAY):
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            return None
        df = normalize_df(df)

        # INI FIX-NYA: intraday biasanya tz-aware (UTC) → bikin naive supaya bisa join ke daily
        if df.index.tz is not None:
            try:
                df.index = df.index.tz_convert(None)
            except Exception:
                df.index = df.index.tz_localize(None)

        return df.dropna(how="any")
    except Exception:
        # diem aja, jangan print apapun
        return None


# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, period: int = 20):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std_factor: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    return upper, lower, ma


def atr(df: pd.DataFrame, period: int = 14):
    high = get_col(df, "High")
    low = get_col(df, "Low")
    close = get_col(df, "Close")

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv(df: pd.DataFrame):
    close = get_col(df, "Close")
    volume = get_col(df, "Volume").fillna(0)

    obv_vals = [0]
    for i in range(1, len(df)):
        c_now = close.iloc[i]
        c_prev = close.iloc[i - 1]
        v_now = volume.iloc[i]

        if c_now > c_prev:
            obv_vals.append(obv_vals[-1] + v_now)
        elif c_now < c_prev:
            obv_vals.append(obv_vals[-1] - v_now)
        else:
            obv_vals.append(obv_vals[-1])
    return pd.Series(obv_vals, index=df.index)


# =========================
# DUMMY FOREIGN
# =========================
def get_dummy_foreign_flow(df: pd.DataFrame):
    close = get_col(df, "Close")
    open_ = get_col(df, "Open")
    volume = get_col(df, "Volume")

    ff = pd.Series(0.0, index=df.index)
    if len(df) < 11:
        return ff

    avg10 = volume.rolling(10).mean()
    green = close > open_
    cond = (volume > avg10) & green
    ff.loc[cond] = volume.loc[cond] * 0.2
    return ff


# =========================
# FEATURE PER SAHAM
# =========================
def build_features_for_symbol(symbol: str):
    # 1) DAILY
    daily = download_daily(symbol)
    if (daily is None) or daily.empty or (len(daily) < 40):
        return None

    close = get_col(daily, "Close")
    open_ = get_col(daily, "Open")
    high = get_col(daily, "High")
    low = get_col(daily, "Low")
    volume = get_col(daily, "Volume")

    # indikator daily
    daily["EMA5"] = ema(close, 5)
    daily["EMA20"] = ema(close, 20)
    daily["EMA50"] = ema(close, 50)
    daily["RSI14"] = rsi(close, 14)
    daily["ATR14"] = atr(daily, 14)

    upper, lower, mid = bollinger_bands(close, 20, 2.0)
    daily["BB_upper"] = upper
    daily["BB_lower"] = lower
    daily["BB_mid"] = mid
    daily["BB_width"] = (upper - lower) / close

    daily["OBV"] = obv(daily)
    daily["OBV_slope"] = daily["OBV"].diff()

    daily["vol_avg_10"] = volume.rolling(10).mean()
    daily["volume_ratio"] = volume / daily["vol_avg_10"]
    daily["ATR_ratio"] = daily["ATR14"] / close

    # === LIQUIDITY METRIC ===
    daily["value_traded"] = close * volume
    daily["liq_score"] = daily["value_traded"].rolling(10).mean()

    # flag likuiditas
    # threshold bisa kamu sesuaikan: 2e10 = Rp20 Miliar nilai transaksi rata2 10 hari
    daily["is_liquid"] = (daily["liq_score"] > LIQUIDITY).astype(int)

    # candle shape
    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    daily["body_ratio"] = body / rng
    daily["close_pos"] = (close - low) / (high - low).replace(0, np.nan)

    # dummy foreign
    daily["foreign_net"] = get_dummy_foreign_flow(daily)
    daily["foreign_pos"] = daily["foreign_net"] / volume.replace(0, np.nan)

    # 2) INTRADAY 1H
    intraday = download_intraday(symbol)
    if intraday is not None and not intraday.empty:
        close_h1 = get_col(intraday, "Close")
        intraday["EMA_H1_20"] = ema(close_h1, 20)

        # resample ke harian (index sudah naive, jadi aman)
        intra_daily = intraday.resample("1D").last()[["Close", "EMA_H1_20"]]
        intra_daily = intra_daily.rename(columns={"Close": "Close_H1"})

        # daily index juga sudah naive tadi → sekarang join
        daily = daily.join(intra_daily, how="left")

        close_h1_d = get_col(daily, "Close_H1")
        ema_h1_20_d = get_col(daily, "EMA_H1_20")
        close_d1 = get_col(daily, "Close")
        ema20_d1 = daily["EMA20"]

        tf_cond = (
            (close_h1_d > ema_h1_20_d) &
            (close_d1 > ema20_d1)
        )
        daily["TF_confirm"] = tf_cond.fillna(False).astype(int)
    else:
        daily["TF_confirm"] = 0

    # 3) label breakout
    daily["Close_next"] = close.shift(-1)
    daily["label_breakout"] = (daily["Close_next"] > high).astype(int)

    daily["symbol"] = symbol
    return daily


# =========================
# BUILD DATASET
# =========================
def build_dataset(universe):
    dfs = []
    for sym in universe:
        try:
            df_sym = build_features_for_symbol(sym)
            if df_sym is not None:
                dfs.append(df_sym)
            #     print(f"[OK] {sym} {len(df_sym)} rows")
            # else:
            #     print(f"[WARN] {sym} no data")
        except Exception as e:
            # print(f"[ERR] {sym}: {e}")
            continue

    if not dfs:
        return None

    all_df = pd.concat(dfs, axis=0)

    need_cols = [
        "volume_ratio","BB_width","RSI14","ATR_ratio",
        "OBV_slope","body_ratio","close_pos","foreign_pos",
        "TF_confirm","label_breakout","is_liquid"
    ]
    all_df = all_df.dropna(subset=need_cols, how="any")
    return all_df


# =========================
# TRAIN MODEL
# =========================
def train_model(df: pd.DataFrame):
    feature_cols = [
        "volume_ratio",
        "BB_width",
        "RSI14",
        "OBV_slope",
        "ATR_ratio",
        "body_ratio",
        "close_pos",
        "foreign_pos",
        "TF_confirm",
    ]
    df_train = df.dropna(subset=feature_cols + ["label_breakout"])
    X = df_train[feature_cols].values
    y = df_train["label_breakout"].values

    split = int(len(df_train) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # coba xgboost
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    print("[INFO] Using XGBClassifier")


    from sklearn.metrics import accuracy_score, f1_score
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"[EVAL] acc={acc:.3f} f1={f1:.3f}")

    return model, feature_cols


# =========================
# HYBRID SCORE
# =========================
def hybrid_score(row: pd.Series, ml_prob: float):
    volume_ratio = float(row["volume_ratio"])
    bb_width = float(row["BB_width"])
    foreign_pos = float(row["foreign_pos"])
    tf_confirm = int(row["TF_confirm"])

    vol_spike = 1 if volume_ratio > 2.5 else 0
    squeeze = 1 if bb_width < 0.05 else 0
    foreign_accum = 1 if foreign_pos > 0.15 else 0

    score = (
        0.33 * vol_spike +
        0.49 * ml_prob +
        0.18 * tf_confirm
    )
    
    return score, vol_spike, squeeze, foreign_accum, tf_confirm


# =========================
# MAIN
# =========================
def main():
    print("Building dataset...")
    df_all = build_dataset(UNIVERSE)
    if df_all is None or df_all.empty:
        print("No data, exit.")
        return

    print("Training model...")
    model, feature_cols = train_model(df_all)

    latest = (
        df_all
        .sort_index()
        .groupby("symbol")
        .tail(1)
        .reset_index(drop=True)
    )

    results = []
    for _, row in latest.iterrows():
        X_row = row[feature_cols].values.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            ml_prob = float(model.predict_proba(X_row)[0, 1])
        else:
            ml_prob = float(model.predict(X_row)[0])

        score, vol_spike, squeeze, foreign_accum, tf_confirm = hybrid_score(row, ml_prob)

        results.append({
            "symbol": row["symbol"],
            "close": float(row["Close"]),
            "score": float(score),
            "ml_prob": ml_prob,
            "vol_spike": vol_spike,
            "squeeze": squeeze,
            "foreign_accum": foreign_accum,
            "tf_confirm": tf_confirm,
            "volume_ratio": float(row["volume_ratio"]),
            "bb_width": float(row["BB_width"]),
            "is_liquid": row["is_liquid"],  # <— ini yang tadi belum ada
        })

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)

    print("\n=== HYBRID BREAKOUT PREDICTION (DAILY + 1H) ===")
    for _, r in df_res.iterrows():
        flag = "🔥" if r["score"] >= 0.7 else ("🟡" if r["score"] >= 0.5 else "⚪")
        print(
            f"{flag} {r['symbol']:8s} | score={r['score']:.2f} | ml={r['ml_prob']:.2f} "
            f"| volx={r['volume_ratio']:.2f} | sqz={r['squeeze']} | ff={r['foreign_accum']} "
            f"| tf={r['tf_confirm']} | liq={'✅' if r['is_liquid'] == 1 else '❌'}"
        )

    print("""
    ------------------------------------------
    LEGEND:
    🔥  Strong Breakout Candidate (score ≥ 0.7)
    🟡  Moderate Setup (0.5 ≤ score < 0.7)
    ⚪  Neutral / Low Confidence (score < 0.5)

    Kolom:
    - score: gabungan semua faktor (0–1)
    - ml: probabilitas model AI harga besok tembus high hari ini
    - volx: rasio volume hari ini vs rata-rata 10 hari
    - sqz: 1 kalau harga lagi dalam Bollinger squeeze (volatilitas rendah)
    - ff: 1 kalau ada indikasi akumulasi asing/bandar (dummy)
    - tf: 1 kalau trend 1H dan 1D sama-sama bullish
    ------------------------------------------
    """)

    print("\nNote:")
    print("- timezone intraday sudah di-drop, jadi join aman")
    print("- kalau intraday kosong, TF_confirm=0 tapi script tetap jalan")


if __name__ == "__main__":
    main()
