import time
import datetime as dt
from typing import List, Dict, Optional

import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 0. SETTING
# ==========================================

# isi dengan saham yang mau kamu pantau
UNIVERSE = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK",
    "AMMN.JK", "ADMR.JK", "BREN.JK", "MEDC.JK",
    "TLKM.JK", "ISAT.JK", "EXCL.JK",
]

INTERVAL = "1m"     # candle 1 menit
PERIOD = "30m"      # ambil 30 menit terakhir (cukup untuk 16 candle ke atas)
LOOKBACK = 15       # rata-rata 15 candle sebelumnya
FACTOR = 3.0        # spike jika last_volume >= FACTOR * avg_prev_15
SLEEP_SEC = 60      # cek tiap 1 menit


# ==========================================
# 1. HELPER
# ==========================================

def fetch_intraday(symbol: str,
                   period: str = PERIOD,
                   interval: str = INTERVAL) -> pd.DataFrame:
    """
    Ambil data intraday per saham.
    """
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return df

    # buang timezone biar simple
    try:
        df = df.tz_localize(None)
    except Exception:
        pass

    # jaga2 volume 0
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    return df


def detect_volume_spike(df: pd.DataFrame,
                        symbol: str,
                        lookback: int = LOOKBACK,
                        factor: float = FACTOR) -> Optional[Dict]:
    """
    Cek apakah candle terakhir punya volume >= factor * rata2 volume
    15 candle sebelumnya.
    """
    if df.empty or "Volume" not in df.columns:
        return None

    df = df.sort_index()
    if len(df) < lookback + 1:
        return None

    vols = df["Volume"].astype(float)
    window = vols.iloc[-(lookback + 1):]  # 15 + 1 (candle terakhir)
    prev = window.iloc[:-1]
    last = window.iloc[-1]

    avg_prev = prev.mean()
    if avg_prev <= 0:
        return None

    ratio = last / avg_prev
    if ratio < factor:
        return None

    last_row = df.iloc[-1]
    ts = df.index[-1]

    return {
        "symbol": symbol,
        "time": ts,
        "last_close": float(last_row.get("Close", np.nan)),
        "last_volume": float(last),
        "avg_prev": float(avg_prev),
        "ratio": float(ratio),
    }


def format_spike_row(spike: Dict) -> str:
    t = spike["time"]
    # format waktu biar enak dibaca
    if isinstance(t, (pd.Timestamp, dt.datetime)):
        t_str = t.strftime("%Y-%m-%d %H:%M")
    else:
        t_str = str(t)
    return (
        f"{spike['symbol']:8} | "
        f"{t_str} | "
        f"Close: {spike['last_close']:10.2f} | "
        f"Vol: {spike['last_volume']:10.0f} | "
        f"Avg15: {spike['avg_prev']:10.0f} | "
        f"Ratio: {spike['ratio']:4.2f}x"
    )


# ==========================================
# 2. SCAN SEKALI
# ==========================================

def scan_once(symbols: List[str]):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Scan volume spike (>= {FACTOR}x, lookback={LOOKBACK})")
    print("-" * 80)

    spikes: List[Dict] = []

    for sym in symbols:
        try:
            df = fetch_intraday(sym)
            if df.empty:
                print(f"{sym:8} -> no data")
                continue

            spike = detect_volume_spike(df, sym)
            if spike:
                spikes.append(spike)
                print("ALERT:", format_spike_row(spike))
            else:
                print(f"{sym:8} -> no spike")
        except Exception as e:
            print(f"{sym:8} -> ERROR: {e}")

    if not spikes:
        print(">> Tidak ada spike volume >= 3x dari rata2 15 candle terakhir.")
    else:
        print("\n=== SUMMARY SPIKES ===")
        for s in spikes:
            print(format_spike_row(s))


# ==========================================
# 3. LOOP TIAP MENIT
# ==========================================

def main_loop():
    print("Start volume spike watcher...")
    print(f"Symbols: {', '.join(UNIVERSE)}")
    print(f"Interval: {INTERVAL}, period: {PERIOD}, check every {SLEEP_SEC}s")
    while True:
        scan_once(UNIVERSE)
        # tidur 1 menit
        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    # kalau mau tes sekali saja, pakai:
    # scan_once(UNIVERSE)
    # kalau mau benar2 jalan terus tiap menit:
    main_loop()
