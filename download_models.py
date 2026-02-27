#!/usr/bin/env python3
"""
download_models.py
==================
Auto-downloads pre-trained model weights from Google Drive.
Run this ONCE after cloning the repo — no training required!

Usage:
    python download_models.py
"""

import os
import sys
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE DOWNLOAD LINKS
# Replace each FILE_ID below with the actual Google Drive file ID.
#
# How to get the file ID from a Google Drive share link:
#   Share link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
#   Copy just the FILE_ID part.
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "facial_emotion_model.h5": {
        "file_id": "165hlZM1P13lBAGz9Xs5pzG6NDotDep2-",
        "dest": "saved_models/facial_emotion_model.h5",
        "size": "~43 MB",
        "description": "Facial Emotion CNN (MiniXception)"
    },
    "speech_emotion_model.h5": {
        "file_id": "15C37uhBjGhmkU_YB-hoUHSmMV_yQiz54",
        "dest": "saved_models/speech_emotion_model.h5",
        "size": "~6 MB",
        "description": "Speech Emotion Attention-BiLSTM"
    },
    "text_bert_model.zip": {
        "file_id": "1pbXigsXEK5Gh58KblhWJHSf78l4hRKo0",
        "dest": "saved_models/text_bert_model.zip",
        "size": "~438 MB",
        "description": "Text Emotion BERT (fine-tuned)",
        "is_zip": True,
        "extract_to": "saved_models/text_bert_model"
    },
}


def download_from_gdrive(file_id: str, dest_path: str, description: str, size: str):
    """Download a file from Google Drive using its file ID."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"  ✅ Already exists: {dest_path} — skipping.")
        return True

    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    print(f"  ⬇️  Downloading {description} ({size})...")
    print(f"      From: {url}")
    print(f"      To:   {dest_path}")

    try:
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
                print(f"\r      [{bar}] {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        print()  # newline after progress bar
        print(f"  ✅ Downloaded: {dest_path}")
        return True

    except Exception as e:
        print(f"\n  ❌ Failed to download {description}: {e}")
        print(f"     Please download manually from:")
        print(f"     https://drive.google.com/file/d/{file_id}/view")
        return False


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file and remove it after extraction."""
    import zipfile
    print(f"  📦 Extracting {zip_path} → {extract_to} ...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  ✅ Extracted to: {extract_to}")


def check_placeholders():
    """Warn if file IDs haven't been filled in."""
    placeholders = [
        name for name, info in MODELS.items()
        if "YOUR_" in info["file_id"]
    ]
    if placeholders:
        print("\n⚠️  WARNING: The following models have placeholder file IDs:")
        for name in placeholders:
            print(f"   - {name}")
        print("\n   Please open download_models.py and fill in the correct")
        print("   Google Drive file IDs before running this script.\n")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  🤖 Multimodal Emotion Recognition — Model Downloader")
    print("=" * 60)
    print()

    check_placeholders()

    success_all = True

    for model_name, info in MODELS.items():
        print(f"\n📥 {info['description']}")
        ok = download_from_gdrive(
            file_id=info["file_id"],
            dest_path=info["dest"],
            description=info["description"],
            size=info["size"]
        )

        if ok and info.get("is_zip") and os.path.exists(info["dest"]):
            extract_zip(info["dest"], info["extract_to"])

        if not ok:
            success_all = False

    print()
    print("=" * 60)
    if success_all:
        print("  ✅ All models downloaded successfully!")
        print("  🚀 You can now run:  python run_dashboard.py")
    else:
        print("  ⚠️  Some models failed. Check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
