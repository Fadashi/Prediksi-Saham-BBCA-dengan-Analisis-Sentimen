"""Modul crawler data stream diskusi Stockbit saham BBCA.

Dilengkapi fitur rate-limiting, deduplikasi, pagination exodus v3 API, dan anonimisasi data.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import os
import time
import requests
from dotenv import load_dotenv

from src.config import CONFIG, RAW_DATA_DIR, set_seed

load_dotenv(BASE_DIR / ".env")


class StockbitStreamCrawler:
    BASE_URL = "https://exodus.stockbit.com/stream/v3/symbol"

    def __init__(
        self,
        symbol: str = None,
        output_file: Path = None,
        bearer_token: str = None,
    ):
        self.symbol = symbol or CONFIG["symbol"]
        self.output_file = output_file or (RAW_DATA_DIR / "stream_bbca.jsonl")
        self.bearer_token = bearer_token or os.getenv("STOCKBIT_BEARER_TOKEN", "")

        self.seen_ids = set()
        self._load_existing_ids()

    def _load_existing_ids(self):
        """Memuat ID postingan yang sudah tersimpan untuk mencegah duplikasi."""
        self.min_stream_id = None
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            sid = str(item.get("id") or item.get("stream_id", ""))
                            if sid:
                                self.seen_ids.add(sid)
                                try:
                                    num_id = int(sid)
                                    if self.min_stream_id is None or num_id < self.min_stream_id:
                                        self.min_stream_id = num_id
                                except ValueError:
                                    pass
                        except Exception:
                            continue
            print(
                f"[Crawler] Ditemukan {len(self.seen_ids)} postingan unik sebelumnya di {self.output_file.name} (Min ID: {self.min_stream_id})."
            )

    def _anonymize_post(self, raw_post: dict) -> dict:
        """Ekstraksi field minimal & anonimisasi."""
        sid = str(raw_post.get("stream_id") or raw_post.get("id", ""))
        created_at = str(raw_post.get("created_at") or raw_post.get("time", ""))
        content = str(raw_post.get("content_original") or raw_post.get("content") or raw_post.get("body", ""))
        like_count = int(raw_post.get("total_likes") or raw_post.get("like_count", 0))

        return {
            "id": sid,
            "created_at": created_at,
            "content": content,
            "like_count": like_count,
        }

    def save_posts(self, posts: list):
        """Menyimpan list postingan terstruktur ke file JSONL dengan deduplikasi ketat."""
        new_count = 0
        with open(self.output_file, "a", encoding="utf-8") as f:
            for post in posts:
                cleaned = self._anonymize_post(post)
                if cleaned["id"] and cleaned["id"] not in self.seen_ids:
                    self.seen_ids.add(cleaned["id"])
                    f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    new_count += 1
        return new_count

    def crawl(self, target: int = 5000000, limit: int = 50, delay_seconds: float = 0.05, min_year: int = 2021):
        """Menjalankan crawling stream Stockbit Exodus v3 API secara cepat sampai ke tahun min_year (2021)."""
        if not self.bearer_token or self.bearer_token == "Bearer your_token_here":
            print("[Crawler Warning] TOKEN Stockbit belum diatur di file .env!")
            return

        token = self.bearer_token.strip()
        auth_header = token if token.lower().startswith("bearer ") else f"Bearer {token}"

        headers = {
            "Authorization": auth_header,
            "User-Agent": "Stockbit/4.6.0 (Android)",
            "Accept": "application/json",
        }

        url = f"{self.BASE_URL}/{self.symbol}"
        last_stream_id = (self.min_stream_id - 1) if (self.min_stream_id and self.min_stream_id > 0) else 0
        batch_num = 0

        print(f"[Crawler] Memulai crawling stream {self.symbol} dari Stockbit API (Start ID: {last_stream_id}, Target Tahun: {min_year})...")

        while True:
            params = {
                "category": "STREAM_CATEGORY_ALL",
                "last_stream_id": last_stream_id,
                "limit": limit,
            }

            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)

                if response.status_code == 429:
                    print("[Crawler] Rate limit (429). Menunggu 5 detik...")
                    time.sleep(5)
                    continue

                if response.status_code != 200:
                    print(f"[Crawler Error] Status code {response.status_code}: {response.text[:200]}")
                    break

                data = response.json()
                if "error_type" in data and data.get("error_type"):
                    print(f"[Crawler API Error] {data.get('error_type')}: {data.get('message')}")
                    break

                streams = data.get("data", {}).get("stream", [])
                if not streams:
                    print(f"[Crawler] Stream selesai/habis untuk {self.symbol}.")
                    break

                added = self.save_posts(streams)
                batch_num += 1

                min_id = None
                oldest_date = None
                for s in streams:
                    sid = s.get("stream_id")
                    cdate = str(s.get("created_at", ""))
                    if sid and (min_id is None or sid < min_id):
                        min_id = sid
                        oldest_date = cdate

                if batch_num % 5 == 0:
                    print(f"[Crawler Batch {batch_num}] Total Unik: {len(self.seen_ids):,} | Tanggal Paling Lama: {oldest_date[:10] if oldest_date else '-'}")

                # Cek jika tanggal postingan sudah melampaui batas tahun min_year (2021)
                if oldest_date and len(oldest_date) >= 4:
                    try:
                        year = int(oldest_date[:4])
                        if year < min_year:
                            print(f"\n[Crawler Sukses] Berhasil mencapai batas tanggal {oldest_date} (Tahun < {min_year}). Crawling selesai!")
                            break
                    except ValueError:
                        pass

                if min_id is None:
                    break

                last_stream_id = min_id - 1
                time.sleep(delay_seconds)

            except Exception as e:
                print(f"[Crawler Exception] {e}")
                time.sleep(3)

        print(f"[Crawler Selesai] Total postingan unik tersimpan di {self.output_file}: {len(self.seen_ids):,}")


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    crawler = StockbitStreamCrawler()
    crawler.crawl(target=5000000, limit=50, delay_seconds=0.05, min_year=2021)
