#!/usr/bin/env python3
"""
Integrated Manga Downloader
Downloads manga from database and saves to repo
"""

import subprocess
import time
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path


def load_database(filename):
    """Load manga database from JSON file"""
    if not os.path.exists(filename):
        print(f"❌ Database file not found: {filename}")
        sys.exit(1)
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_manga_from_database(database, index):
    """Get manga entry by index (1-based)"""
    if index < 1 or index > len(database["manga"]):
        print(f"❌ Invalid manga index: {index}")
        print(f"💡 Valid range: 1-{len(database['manga'])}")
        sys.exit(1)
    
    return database["manga"][index - 1]


def setup_downloader():
    """Clone and setup the manga downloader"""
    print("🔧 Setting up downloader...")
    print("=" * 60)
    
    # Clone the repository
    if not os.path.exists("AIO-Webtoon-Downloader"):
        print("📦 Cloning AIO-Webtoon-Downloader repository...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/zzyil/AIO-Webtoon-Downloader.git"
        ], check=True)
        print("✓ Repository cloned!\n")
    else:
        print("✓ Repository already exists!\n")
    
    # Change to the repository directory
    os.chdir("AIO-Webtoon-Downloader")
    
    # Install requirements
    print("📦 Installing dependencies...")
    subprocess.run([
        "pip", "install", "-q", "-r", "requirements.txt"
    ], check=True)
    print("✓ Dependencies installed!\n")
    
    print("=" * 60)
    print()


def download_chapters(url, start, end, manga_title):
    """Download chapters and return path to PDFs"""
    print("=" * 60)
    print(f"📥 Downloading chapters {start}-{end}...")
    print("=" * 60)
    
    cmd = [
        "python3", "aio-dl.py",
        "--chapters", f"{start}-{end}",
        "--format", "pdf",
        "--width", "800",
        "--quality", "70",
        "--keep-chapters",
        "--verbose",
        url
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Download completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user!")
        sys.exit(1)


def organize_downloads(manga_title, output_dir="../downloaded_manga"):
    """Organize downloaded files into output directory"""
    print("\n" + "=" * 60)
    print("📦 Organizing downloaded files...")
    print("=" * 60)
    
    # Go back to parent directory
    os.chdir("..")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Find comics folder from downloader
    comics_path = "AIO-Webtoon-Downloader/comics"
    if not os.path.exists(comics_path):
        print("❌ No comics folder found!")
        return []
    
    # Create manga-specific folder
    safe_title = manga_title.replace(' ', '_').replace('/', '-')
    manga_folder = os.path.join(output_dir, safe_title)
    os.makedirs(manga_folder, exist_ok=True)
    
    # Find and copy all PDFs
    pdf_files = list(Path(comics_path).glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found!")
        return []
    
    print(f"✓ Found {len(pdf_files)} PDF files")
    
    copied_files = []
    for pdf_file in pdf_files:
        dest_path = os.path.join(manga_folder, pdf_file.name)
        subprocess.run(["cp", str(pdf_file), dest_path], check=True)
        copied_files.append(dest_path)
        print(f"  ✓ Copied: {pdf_file.name}")
    
    print(f"\n✅ Files organized in: {manga_folder}")
    print(f"📁 Total files: {len(copied_files)}")
    
    return copied_files


def print_database_stats(database):
    """Print database statistics"""
    meta = database["metadata"]
    print("\n" + "=" * 60)
    print("📊 DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total Manga: {meta['total_manga']}")
    print(f"Completed: {meta['completed_count']} ({meta['completed_count']/meta['total_manga']*100:.1f}%)")
    print(f"Pending: {meta['pending_count']} ({meta['pending_count']/meta['total_manga']*100:.1f}%)")
    print("=" * 60 + "\n")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Download manga from database and save to repo'
    )
    
    parser.add_argument('--database', type=str, required=True,
                       help='Path to manga database JSON file')
    parser.add_argument('--manga-index', type=int, required=True,
                       help='Which manga to download (1-based index)')
    parser.add_argument('--start-chapter', type=int, default=1,
                       help='Starting chapter number')
    parser.add_argument('--end-chapter', type=int, required=True,
                       help='Ending chapter number')
    parser.add_argument('--output-dir', type=str, default='downloaded_manga',
                       help='Output directory for PDFs')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 Integrated Manga Downloader")
    print("=" * 60)
    print()
    
    # Load database
    print(f"📖 Loading database: {args.database}")
    database = load_database(args.database)
    print_database_stats(database)
    
    # Get manga from database
    manga = get_manga_from_database(database, args.manga_index)
    
    print("=" * 60)
    print("📚 Selected Manga")
    print("=" * 60)
    print(f"Index: #{manga['id']}")
    print(f"Title: {manga['title']}")
    print(f"URL: {manga['url']}")
    print(f"Status: {'✅ Completed' if manga['completed'] else '⏳ Pending'}")
    print("=" * 60)
    print()
    
    # Setup downloader
    setup_downloader()
    
    # Download chapters
    print("=" * 60)
    print("📥 Download Configuration")
    print("=" * 60)
    print(f"📚 Chapters: {args.start_chapter}-{args.end_chapter}")
    print(f"📦 Total: {args.end_chapter - args.start_chapter + 1} chapters")
    print("=" * 60)
    print()
    
    success = download_chapters(
        url=manga['url'],
        start=args.start_chapter,
        end=args.end_chapter,
        manga_title=manga['title']
    )
    
    if not success:
        print("\n❌ Download failed!")
        sys.exit(1)
    
    # Organize downloads
    pdf_files = organize_downloads(manga['title'], args.output_dir)
    
    if not pdf_files:
        print("\n❌ No files were downloaded!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Download Complete!")
    print("=" * 60)
    print(f"✅ Downloaded {len(pdf_files)} PDF files")
    print(f"📁 Location: {args.output_dir}")
    print("=" * 60)
    
    # Print file paths for next step
    print("\n📄 Downloaded PDFs:")
    for pdf in pdf_files:
        print(f"  • {pdf}")


if __name__ == "__main__":
    main()