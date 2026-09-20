"""
Automated downloader & ingestion pipeline for real-world floorplan corpora:
- RPLAN: Real residential floorplans (Wu et al., ACM TOG 2019)
- ResPlan: Vector residential floorplans (Abouagour et al., 2025)
Provides automated fetching, zip extraction, and canonical FloorGen vector parsing into data/processed.
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from typing import Optional

from floorgen.data.scripts.parse_rplan import ingest_rplan_source, init_sqlite_db

RPLAN_MIRROR_URL = "https://huggingface.co/datasets/metindeder/rplan-floorplan-edited/resolve/main/rplan_dataset.zip"
RPLAN_REPO_URL = "https://github.com/nero-science/RPLAN"
RESPLAN_REPO_URL = "https://github.com/nero-science/ResPlan"


def download_file(url: str, dest_path: str):
    """Downloads a file with a visual terminal progress bar."""
    print(f"[FloorGen Downloader] Downloading from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.1f}MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\n[FloorGen Downloader] Download failed: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str):
    """Unpacks zip archives into destination directory."""
    print(f"[FloorGen Downloader] Extracting {archive_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    print("Extraction complete.")


def ingest_rplan_dataset(raw_source: str, output_dir: str = "data/processed", max_samples: int = 5000):
    """
    Parses RPLAN plans into FloorGen canonical format.
    """
    return ingest_rplan_source(raw_source, output_dir, max_samples=max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloorGen Dataset Downloader & Ingestor")
    parser.add_argument("--dataset", type=str, choices=["rplan", "resplan"], default="rplan")
    parser.add_argument("--dest_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)
    zip_dest = os.path.join(args.dest_dir, "rplan_dataset.zip")
    
    if not os.path.exists(zip_dest):
        print(f"[FloorGen Downloader] Fetching real RPLAN dataset archive...")
        success = download_file(RPLAN_MIRROR_URL, zip_dest)
        if not success:
            print(f"[FloorGen Downloader] Please manually place the dataset archive in {zip_dest}")
            sys.exit(1)
    else:
        print(f"[FloorGen Downloader] Found existing RPLAN dataset archive at {zip_dest}")

    # Ingest into canonical format
    ingest_rplan_source(zip_dest, output_dir=args.output_dir, max_samples=args.max_samples)
