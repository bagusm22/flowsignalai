import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings

# biar gak banyak bacot
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# KONFIG
# =========================
SYMBOLS = [
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
INTERVAL = "5m"
REFRESH_SECONDS = 300
SPIKE_THRESHOLD = 3.0  # 3x rata-rata 5m

# =========================
# HELPER
# =========================
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
        auto_adjust=False,
        group_by="column",  # << penting banget
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


def avg_5m_volume(symbol: str, lookback_days: int = 5) -> float:
    raw = download_5m(symbol, period=f"{lookback_days}d")
    if isinstance(raw, tuple):
        df = raw[0]
    else:
        df = raw
    if df.empty:
        print(f"⚠️ {symbol} 5m historis kosong (yahoo ga ada).")
        return np.nan
    df = _normalize_df(df)
    # Pastikan kolom Volume ada
    vol_col = next((c for c in df.columns if "Volume" in c or "volume" in c), None)
    if vol_col is None:
        print(f"⚠️ {symbol} tidak ada kolom Volume (mungkin suspend / illiquid).")
        return np.nan

    df = df[df[vol_col] > 0]
    if df.empty:
        return np.nan

    return float(df[vol_col].mean())



def check_realtime_volume(symbol: str, avg_5m: float):
    """
    Ambil candle 5 menit TERAKHIR hari ini, lalu bandingkan dengan rata-rata 5m historis.
    """
    df = download_5m(symbol, period="1d")
    if df.empty:
        print(f"⚠️ {symbol} 5m hari ini kosong, skip.")
        return None

    # sekarang df harus punya kolom 'Volume' dan 'Close'
    cols = [c.lower() for c in df.columns]
    has_vol = any("volume" in c for c in cols)
    has_close = any(c == "close" for c in cols) or any("close" in c for c in cols)
    if not has_vol or not has_close:
        print(f"⚠️ {symbol} tidak punya kolom Volume atau Close.")
        return None

    # aman ambil baris terakhir
    last = df.iloc[-1]

    vol_now = float(last[[c for c in df.columns if "Volume" in c][0]])
    close_now = float(last[[c for c in df.columns if "Close" in c][0]])

    if avg_5m is None or not np.isfinite(avg_5m) or avg_5m <= 0:
        ratio = 0.0
    else:
        ratio = float(vol_now / avg_5m)

    return {
        "symbol": symbol,
        "time": df.index[-1],
        "close": close_now,
        "vol_now": vol_now,
        "avg_5m": float(avg_5m) if avg_5m else 0.0,
        "ratio": ratio,
    }

# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    print("🚀 Realtime 5m Volume Spike Monitor")
    print("===================================")

    # pre-hitungan baseline
    print("⏳ hitung baseline rata-rata volume 5m ...")
    avg_vol_dict = {}
    for s in SYMOLS if False else SYMBOLS:  # placeholder kalau mau filter
        avg_vol_dict[s] = avg_5m_volume(s, 1)

    while True:
        print(f"\n🕐 Cek pada {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = []  # tampung semua hasil

        for sym in SYMBOLS:
            baseline = avg_vol_dict.get(sym)
            data = check_realtime_volume(sym, baseline)
            if data is None:
                continue
            results.append(data)

        # === SORT DARI RATIO TERTINGGI ===
        results = sorted(results, key=lambda x: x["ratio"], reverse=True)

        # === TAMPILKAN HASIL ===
        for data in results:
            sym = data["symbol"]
            if data["ratio"] >= SPIKE_THRESHOLD:
                print(
                    f"🔥 {sym} | {data['time']} | Close={data['close']:.2f} | "
                    f"VolNow={data['vol_now']:.0f} | Avg5m={data['avg_5m']:.0f} | "
                    f"Ratio={data['ratio']:.2f}x"
                )
            else:
                print(
                    f"{sym:8} | Ratio={data['ratio']:.2f}x | Vol={data['vol_now']:.0f} | Close={data['close']:.2f}"
                )

        print("⏳ tunggu 5 menit...\n")
        time.sleep(REFRESH_SECONDS)
