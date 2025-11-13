# -*- coding: utf-8 -*-
"""
Build UNIVERSE (IDX tickers '.JK') dari sumber statis (tanpa JS):
- Wikipedia: IDX Composite (statis)
- MarketScreener: daftar emiten Indonesia (paginated, statis)

Dibuat tahan LibreSSL:
- requests selalu verify ke certifi
- pakai retry
"""

import re, html, time, sys, traceback
from typing import Set, List
import requests, certifi, pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh) UNIVERSEFetcher/1.0"}

WIKI_URL = "https://en.wikipedia.org/wiki/IDX_Composite"
MS_PAGES = [f"https://www.marketscreener.com/quote/country/indonesia-141/page-{i}/" for i in range(1, 11)]
MS_PAGES[0] = "https://www.marketscreener.com/quote/country/indonesia-141/"

BLACKLIST = {
    "IDX","IHSG","JCI","JKSE","USD","PT","TBK","HTML","HTTP","HTTPS",
    "PDF","FAQ","JAKARTA","MBX","DBX","SECTOR","INDONESIA","ETFs","ETF"
}

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def extract_symbols_from_text(text: str) -> Set[str]:
    # ambil kapital 3–5 huruf (BBCA, TLKM, AMMN, MDKA, dst.)
    cands = set(re.findall(r"(?<![A-Z])([A-Z]{3,5})(?![A-Z])", text))
    return {c for c in cands if c not in BLACKLIST and 3 <= len(c) <= 5}

def fetch_html(session: requests.Session, url: str, allow_insecure=False) -> str:
    try:
        r = session.get(url, timeout=25, verify=certifi.where())
        r.raise_for_status()
        return html.unescape(r.text)
    except Exception as e:
        if allow_insecure:
            # LAST RESORT (tidak disarankan): matikan verify
            try:
                r = session.get(url, timeout=25, verify=False)
                r.raise_for_status()
                return html.unescape(r.text)
            except Exception:
                raise e
        raise e

def fetch_wiki_symbols(session: requests.Session, allow_insecure=False) -> Set[str]:
    syms = set()
    text = fetch_html(session, WIKI_URL, allow_insecure=allow_insecure)
    # coba parse tabel via pandas dari string html (bukan URL langsung)
    try:
        tables = pd.read_html(text)
        for df in tables:
            for col in df.columns:
                series = df[col].astype(str)
                for val in series:
                    syms |= extract_symbols_from_text(val)
    except Exception:
        pass
    # tambahan regex langsung dari html
    syms |= extract_symbols_from_text(text)
    return syms

def fetch_marketscreener_symbols(session: requests.Session, allow_insecure=False) -> Set[str]:
    syms = set()
    for url in MS_PAGES:
        try:
            text = fetch_html(session, url, allow_insecure=allow_insecure)
            syms |= extract_symbols_from_text(text)
            time.sleep(0.4)
        except Exception:
            continue
    return syms

def attach_jk(symbols: Set[str]) -> List[str]:
    return [f"{s}.JK" for s in sorted(symbols)]

def main():
    s = make_session()
    try:
        # 1) coba normal (verify=certifi)
        w = fetch_wiki_symbols(s, allow_insecure=False)
        m = fetch_marketscreener_symbols(s, allow_insecure=False)
        merged = w | m
        # buang beberapa noise umum
        noise = {"INDO","TAMA","JAVA","BANK","MEDIA","MINER","FOOD","AUTO","KOTA","JAYA"}
        merged = {x for x in merged if x not in noise}

        if len(merged) < 300:
            print(f"[WARN] Baru terkumpul {len(merged)} symbols. Coba fallback insecure verify...")
            # 2) fallback insecure (tidak disarankan, tapi bantu environment lama)
            w2 = fetch_wiki_symbols(s, allow_insecure=True)
            m2 = fetch_marketscreener_symbols(s, allow_insecure=True)
            merged |= (w2 | m2)
            merged = {x for x in merged if x not in noise}

        UNIVERSE = attach_jk(merged)
        print(f"# Total tickers: {len(UNIVERSE)}")
        print("UNIVERSE = [")
        for i, t in enumerate(UNIVERSE, 1):
            end = " " if (i % 12 == 0) else ""
            print(f'    "{t}",{end}')
        print("]")

    except Exception as e:
        print("[ERROR] Gagal membangun UNIVERSE:", e, file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()
