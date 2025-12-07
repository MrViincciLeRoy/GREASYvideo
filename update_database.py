#!/usr/bin/env python3
"""
Update Manga Database
Marks manga as completed after processing
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


def load_database(filename):
    """Load manga database from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_database(database, filename):
    """Save manga database to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)


def mark_as_complete(database, manga_index, video_path=None, notes=""):
    """Mark a manga as completed"""
    if manga_index < 1 or manga_index > len(database["manga"]):
        print(f"❌ Invalid manga index: {manga_index}")
        return False
    
    manga = database["manga"][manga_index - 1]
    
    # Update manga entry
    manga["completed"] = True
    manga["processed_at"] = datetime.now().isoformat()
    
    if video_path:
        manga["video_path"] = video_path
    
    if notes:
        manga["notes"] = notes
    
    # Update metadata counts
    if not manga.get("completed"):  # Only update if wasn't completed before
        database["metadata"]["completed_count"] += 1
        database["metadata"]["pending_count"] = max(0, database["metadata"]["pending_count"] - 1)
    
    return True


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Update manga database after processing'
    )
    
    parser.add_argument('--database', type=str, required=True,
                       help='Path to manga database JSON file')
    parser.add_argument('--manga-index', type=int, required=True,
                       help='Which manga to mark as complete (1-based index)')
    parser.add_argument('--completed', action='store_true',
                       help='Mark manga as completed')
    parser.add_argument('--video-path', type=str, default=None,
                       help='Path to generated video')
    parser.add_argument('--notes', type=str, default="",
                       help='Additional notes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📝 Database Update Tool")
    print("=" * 60)
    print()
    
    # Load database
    print(f"📖 Loading database: {args.database}")
    database = load_database(args.database)
    
    # Get manga info
    if args.manga_index < 1 or args.manga_index > len(database["manga"]):
        print(f"❌ Invalid manga index: {args.manga_index}")
        return
    
    manga = database["manga"][args.manga_index - 1]
    print(f"📚 Manga: {manga['title']}")
    print(f"   Current status: {'✅ Completed' if manga['completed'] else '⏳ Pending'}")
    print()
    
    if args.completed:
        print("🔄 Marking as completed...")
        
        success = mark_as_complete(
            database,
            args.manga_index,
            video_path=args.video_path,
            notes=args.notes
        )
        
        if success:
            # Save database
            save_database(database, args.database)
            
            print("✅ Database updated!")
            print(f"   Status: {'✅ Completed' if manga['completed'] else '⏳ Pending'}")
            if args.video_path:
                print(f"   Video: {args.video_path}")
            if args.notes:
                print(f"   Notes: {args.notes}")
            print()
            
            # Print stats
            meta = database["metadata"]
            print("📊 Updated Statistics:")
            print(f"   Total: {meta['total_manga']}")
            print(f"   Completed: {meta['completed_count']} ({meta['completed_count']/meta['total_manga']*100:.1f}%)")
            print(f"   Pending: {meta['pending_count']} ({meta['pending_count']/meta['total_manga']*100:.1f}%)")
        else:
            print("❌ Failed to update database")
    else:
        print("ℹ️  No changes made (use --completed flag to mark as complete)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()