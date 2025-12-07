
#!/usr/bin/env python3
"""
Google Colab Manga Batch Downloader - Database Edition
Downloads manga from database and tracks completion
"""

import subprocess
import time
import sys
import os
import json
from datetime import datetime

# ==================== CONFIGURATION ====================
DATABASE_FILE = "data/manga_database_20251207_061922.json"  # Your database file
MANGA_INDEX = 2  # Which manga to download (1-based index)
TOTAL_CHAPTERS = 10  # Number of chapters to download
BATCH_SIZE = 2  # Chapters per batch
START_CHAPTER = 1  # Starting chapter number
# =======================================================

def load_database(filename):
    """Load manga database from JSON file"""
    if not os.path.exists(filename):
        print(f"❌ Database file not found: {filename}")
        print("💡 Make sure you've uploaded your database file to Colab!")
        sys.exit(1)
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_database(database, filename):
    """Save manga database to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

def mark_manga_complete(database, manga_id, chapters_downloaded, download_path):
    """Mark a manga as completed in the database"""
    for manga in database["manga"]:
        if manga["id"] == manga_id:
            manga["completed"] = True
            manga["processed_at"] = datetime.now().isoformat()
            manga["video_path"] = download_path
            manga["notes"] = f"Downloaded {chapters_downloaded} chapters"
            
            # Update metadata
            database["metadata"]["completed_count"] += 1
            database["metadata"]["pending_count"] -= 1
            return True
    return False

def get_manga_from_database(database, index):
    """Get manga entry by index (1-based)"""
    if index < 1 or index > len(database["manga"]):
        print(f"❌ Invalid manga index: {index}")
        print(f"💡 Valid range: 1-{len(database['manga'])}")
        sys.exit(1)
    
    return database["manga"][index - 1]

def setup_environment():
    """Install required packages and clone the repository"""
    print("🔧 Setting up environment...")
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

def run_batch_download(url, start, end, batch_num, total_batches):
    """Run a single batch download"""
    print("=" * 60)
    print(f"📥 Batch {batch_num} of {total_batches}")
    print(f"📖 Downloading chapters {start}-{end}...")
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
        print(f"\n✅ Batch {batch_num} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Batch {batch_num} failed with error code {e.returncode}")
        print(f"💡 Resume from chapter {start} by changing START_CHAPTER variable.")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user!")
        print(f"💡 Resume from chapter {start} by changing START_CHAPTER variable.")
        sys.exit(1)

def download_files_from_colab(manga_title):
    """Package and prepare files for download from Colab"""
    print("\n" + "=" * 60)
    print("📦 Preparing files for download...")
    print("=" * 60)
    
    comics_path = "comics"
    if os.path.exists(comics_path):
        # Count PDF files
        pdf_files = [f for f in os.listdir(comics_path) if f.endswith('.pdf')]
        print(f"✓ Found {len(pdf_files)} PDF files in comics folder")
        
        # Create a zip file with manga title
        safe_title = manga_title.replace(' ', '_').replace('/', '-')
        zip_name = f"{safe_title}_chapters.zip"
        
        print(f"🗜️  Creating zip archive: {zip_name}")
        subprocess.run([
            "zip", "-r", zip_name, comics_path
        ], check=True)
        
        print("✅ Files packaged successfully!")
        print(f"\n💡 To download your files:")
        print(f"   1. Check the file browser on the left (📁)")
        print(f"   2. Find '{zip_name}'")
        print(f"   3. Right-click and select 'Download'")
        print(f"\n   Or run this code to download directly:")
        print(f"   from google.colab import files")
        print(f"   files.download('{zip_name}')")
        
        return zip_name
    else:
        print("⚠️  No comics folder found!")
        return None

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
    print("=" * 60)
    print("🎯 Database-Driven Manga Downloader")
    print("=" * 60)
    print()
    
    # Load database
    print(f"📖 Loading database: {DATABASE_FILE}")
    database = load_database(DATABASE_FILE)
    print_database_stats(database)
    
    # Get manga from database
    manga = get_manga_from_database(database, MANGA_INDEX)
    
    print("=" * 60)
    print("📚 Selected Manga")
    print("=" * 60)
    print(f"Index: #{manga['id']}")
    print(f"Title: {manga['title']}")
    print(f"URL: {manga['url']}")
    print(f"Status: {'✅ Completed' if manga['completed'] else '⏳ Pending'}")
    print("=" * 60)
    print()
    
    # Check if already completed
    if manga['completed']:
        user_input = input("⚠️  This manga is already marked as completed. Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborted.")
            return
    
    # Setup environment
    setup_environment()
    
    # Calculate number of batches
    num_batches = (TOTAL_CHAPTERS + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("=" * 60)
    print("📥 Download Configuration")
    print("=" * 60)
    print(f"📚 Total chapters: {TOTAL_CHAPTERS}")
    print(f"📦 Batch size: {BATCH_SIZE} chapters")
    print(f"🔢 Number of batches: {num_batches}")
    print(f"▶️  Starting from chapter: {START_CHAPTER}")
    print("=" * 60)
    print()
    
    # Loop through batches
    chapters_downloaded = 0
    for i in range(1, num_batches + 1):
        # Calculate chapter range for this batch
        start = (i - 1) * BATCH_SIZE + START_CHAPTER
        end = min(start + BATCH_SIZE - 1, START_CHAPTER + TOTAL_CHAPTERS - 1)
        
        # Run the batch
        success = run_batch_download(manga['url'], start, end, i, num_batches)
        
        if not success:
            print("\n⚠️  Stopping due to error.")
            zip_name = download_files_from_colab(manga['title'])
            
            # Update database with partial completion
            if chapters_downloaded > 0:
                manga["notes"] = f"Partially downloaded {chapters_downloaded} chapters (interrupted)"
                save_database(database, f"../{DATABASE_FILE}")
                print(f"\n💾 Database updated with partial progress")
            
            sys.exit(1)
        
        chapters_downloaded += (end - start + 1)
        
        # Brief pause between batches
        if i < num_batches:
            print(f"⏸️  Pausing for 3 seconds before next batch...")
            time.sleep(3)
            print()
    
    print("=" * 60)
    print("🎉 All downloads complete!")
    print("=" * 60)
    print(f"✅ Downloaded chapters {START_CHAPTER}-{START_CHAPTER + TOTAL_CHAPTERS - 1}")
    print(f"📁 Files are in the 'comics' folder")
    print("=" * 60)
    
    # Prepare files for download
    zip_name = download_files_from_colab(manga['title'])
    
    # Update database
    if zip_name:
        download_path = f"comics/{zip_name}"
        mark_manga_complete(database, manga['id'], chapters_downloaded, download_path)
        
        # Save database (go back to parent directory first)
        os.chdir('..')
        save_database(database, DATABASE_FILE)
        
        print("\n💾 Database updated!")
        print(f"✅ Manga #{manga['id']} marked as complete")
        print_database_stats(database)

if __name__ == "__main__":
    main()