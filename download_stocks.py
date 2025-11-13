import pandas as pd
import json

# sumber resmi BEI (versi Indonesia)
URL_ID = "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham"
# bisa juga pakai yang versi Inggris:
# URL_EN = "https://www.idx.co.id/en/market-data/stocks-data/stock-list"

def fetch_idx_tickers(url: str = URL_ID):
    # baca semua tabel di halaman
    tables = pd.read_html(url)
    if not tables:
        raise RuntimeError("Tidak menemukan tabel di halaman IDX")

    # biasanya tabel pertama yang berisi daftar saham
    df = tables[0]

    # cari kolom yang isinya kode saham
    # di halaman BEI biasanya namanya "Kode" atau "Code"
    possible_cols = ["Kode", "Code", "Stock Code", "Kode Saham"]
    code_col = None
    for c in possible_cols:
        if c in df.columns:
            code_col = c
            break

    if code_col is None:
        # fallback: ambil kolom pertama
        code_col = df.columns[0]

    # ambil kolom kode
    tickers = (
        df[code_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # buang yang kosong
    tickers = [t for t in tickers if t and t != "NAN"]

    # buang duplikat dengan tetap jaga urutan
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    return unique_tickers

if __name__ == "__main__":
    tickers = fetch_idx_tickers()

    # kalau kamu cuma mau JSON murni:
    print(json.dumps(tickers, indent=2, ensure_ascii=False))

    # kalau kamu mau langsung format array python buat dimasukin ke script AI kamu:
    print("\n# Python array / UNIVERSE")
    print("UNIVERSE = [")
    for t in tickers:
        print(f'    "{t}",')
    print("]")
