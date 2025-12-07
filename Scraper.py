
"""
Asura Comics Manga Scraper - Colab Compatible (Fixed Pagination)
Scrapes manga links from asuracomic.net by actually clicking next button
"""

# Installation commands for Google Colab
# Run these first in a Colab cell:
"""
!pip install playwright
!playwright install chromium
!playwright install-deps
"""

import asyncio
from playwright.async_api import async_playwright
import json
from datetime import datetime

async def scrape_asura_manga(start_page=1, max_links=2000):
    """
    Scrape manga links from Asura Comics
    
    Args:
        start_page: Starting page number (will click next to reach it)
        max_links: Maximum number of links to collect
    """
    base_url = "https://asuracomic.net/series?page=1&genres=&status=-1&types=-1&order=bookmarks"
    all_links = set()  # Use set to avoid duplicates
    current_page = 1
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        print(f"Starting scrape...")
        print(f"Target: {max_links} links\n")
        
        try:
            # Go to first page
            print("Loading initial page...")
            await page.goto(base_url, wait_until='domcontentloaded', timeout=60000)
            await asyncio.sleep(3)  # Wait for JS to load
            
            # Navigate to start_page if needed
            if start_page > 1:
                print(f"Navigating to page {start_page}...")
                for i in range(1, start_page):
                    # Try to find and click next button
                    next_btn = await page.query_selector('a[rel="next"]')
                    if not next_btn:
                        # Try alternative selectors
                        next_btn = await page.query_selector('a:has-text("Next")')
                    
                    if next_btn:
                        await next_btn.click()
                        await asyncio.sleep(2)
                        current_page += 1
                        print(f"  Navigated to page {current_page}")
                    else:
                        print(f"  Could not find next button at page {current_page}")
                        break
            
            # Now start collecting links
            print(f"\nStarting collection from page {current_page}...\n")
            
            while len(all_links) < max_links:
                print(f"Scraping page {current_page}...")
                
                # Wait a bit for content to stabilize
                await asyncio.sleep(2)
                
                # Extract manga links - try multiple selectors
                links = await page.evaluate('''() => {
                    const uniqueLinks = new Set();
                    
                    // Try multiple selectors to find manga links
                    const selectors = [
                        'a[href*="/series/"]',
                        '.grid a',
                        '.series-list a',
                        'a[href^="/series"]'
                    ];
                    
                    selectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(link => {
                            const href = link.href;
                            // Only series pages, not chapters
                            if (href && href.includes('/series/') && !href.includes('/chapter')) {
                                uniqueLinks.add(href);
                            }
                        });
                    });
                    
                    return Array.from(uniqueLinks);
                }''')
                
                # Add new links
                old_count = len(all_links)
                all_links.update(links)
                new_count = len(all_links) - old_count
                
                print(f"  Found {new_count} new links (Total: {len(all_links)})")
                
                # Check if we've reached the target
                if len(all_links) >= max_links:
                    print(f"\n✓ Target reached: {len(all_links)} links collected")
                    break
                
                # Find next button with multiple strategies
                next_button = await page.query_selector('a[rel="next"]')
                if not next_button:
                    next_button = await page.query_selector('a:has-text("Next")')
                if not next_button:
                    next_button = await page.query_selector('nav a:last-child')
                
                if not next_button:
                    print("\n✗ No more pages available")
                    break
                
                # Check if next button is disabled
                is_disabled = await page.evaluate('''(btn) => {
                    return btn.classList.contains('disabled') || 
                           btn.hasAttribute('disabled') ||
                           btn.getAttribute('aria-disabled') === 'true';
                }''', next_button)
                
                if is_disabled:
                    print("\n✗ Reached last page (next button disabled)")
                    break
                
                # Click next button
                print(f"  Clicking next button...")
                await next_button.click()
                current_page += 1
                
                # Wait for navigation/content change
                await asyncio.sleep(3)
                
        except Exception as e:
            print(f"\n✗ Error during scraping: {str(e)}")
        
        finally:
            await browser.close()
    
    # Convert set to list and trim to max_links
    all_links = list(all_links)[:max_links]
    
    return all_links, current_page

async def main():
    """Main function to run the scraper"""
    print("="*60)
    print("Asura Comics Manga Scraper (Fixed)")
    print("="*60 + "\n")
    
    # Scrape manga links - start from page 4
    links, last_page = await scrape_asura_manga(start_page=1, max_links=1000)
    
    # Print results
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total links collected: {len(links)}")
    print(f"Last page reached: {last_page}")
    
    # Create database structure with completion tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"manga_database_{timestamp}.json"
    
    # Build database with individual manga entries
    manga_database = {
        "metadata": {
            "created_at": timestamp,
            "total_manga": len(links),
            "last_page_scraped": last_page,
            "completed_count": 0,
            "pending_count": len(links)
        },
        "manga": []
    }
    
    # Add each manga as a database entry
    for idx, link in enumerate(links, 1):
        # Extract manga ID and title from URL
        manga_id = link.split('/')[-1]
        title = manga_id.replace('-', ' ').title()
        
        manga_entry = {
            "id": idx,
            "manga_id": manga_id,
            "title": title,
            "url": link,
            "completed": False,
            "processed_at": None,
            "video_path": None,
            "notes": ""
        }
        manga_database["manga"].append(manga_entry)
    
    # Save database
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(manga_database, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Database saved to: {filename}")
    print(f"  - Total entries: {len(links)}")
    print(f"  - All marked as: Incomplete")
    
    # Display first 10 entries as sample
    print(f"\nSample database entries (first 10):")
    for entry in manga_database["manga"][:10]:
        print(f"  {entry['id']}. {entry['title']}")
        print(f"     URL: {entry['url']}")
        print(f"     Status: {'✓ Complete' if entry['completed'] else '○ Pending'}")
        print()
    
    return links, manga_database

# Helper functions for database management
def load_database(filename):
    """Load manga database from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_database(database, filename):
    """Save manga database to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

def mark_as_complete(database, manga_id, video_path=None, notes=""):
    """Mark a manga as completed"""
    for manga in database["manga"]:
        if manga["manga_id"] == manga_id or manga["id"] == manga_id:
            manga["completed"] = True
            manga["processed_at"] = datetime.now().isoformat()
            if video_path:
                manga["video_path"] = video_path
            if notes:
                manga["notes"] = notes
            
            # Update metadata counts
            database["metadata"]["completed_count"] += 1
            database["metadata"]["pending_count"] -= 1
            return True
    return False

def get_pending_manga(database, limit=None):
    """Get list of manga that haven't been processed yet"""
    pending = [m for m in database["manga"] if not m["completed"]]
    if limit:
        return pending[:limit]
    return pending

def get_completed_manga(database):
    """Get list of manga that have been processed"""
    return [m for m in database["manga"] if m["completed"]]

def print_database_stats(database):
    """Print database statistics"""
    meta = database["metadata"]
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    print(f"Total Manga: {meta['total_manga']}")
    print(f"Completed: {meta['completed_count']} ({meta['completed_count']/meta['total_manga']*100:.1f}%)")
    print(f"Pending: {meta['pending_count']} ({meta['pending_count']/meta['total_manga']*100:.1f}%)")
    print(f"Created: {meta['created_at']}")
    print("="*60)

# Run the scraper
if __name__ == "__main__":
    # For Colab, use this:
    links = await main()
    
    # If running in standard Python (not Colab), use:
    # asyncio.run(main())