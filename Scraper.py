#!/usr/bin/env python3
"""
Asura Comics Manga Scraper with Command Line Arguments
"""

import asyncio
from playwright.async_api import async_playwright
import json
from datetime import datetime
import argparse


async def scrape_asura_manga(start_page=1, max_links=2000):
    """
    Scrape manga links from Asura Comics - Direct URL approach
    """
    all_links = set()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        print(f"Starting scrape...")
        print(f"Target: {max_links} links")
        print(f"Starting from page: {start_page}\n")
        
        try:
            current_page = start_page
            max_page = 100  # Safety limit
            
            while len(all_links) < max_links and current_page <= max_page:
                # Build URL with page number directly
                url = f"https://asuracomic.net/series?page={current_page}&genres=&status=-1&types=-1&order=bookmarks"
                
                print(f"📄 Scraping page {current_page}...")
                print(f"   URL: {url}")
                
                try:
                    # Go directly to the page URL
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    await asyncio.sleep(3)
                    
                    # Wait for content to load
                    await page.wait_for_selector('.grid, .series-list, a[href*="/series/"]', timeout=10000)
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"   ⚠ Error loading page: {e}")
                    break
                
                # Extract manga links
                links = await page.evaluate('''() => {
                    const uniqueLinks = new Set();
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
                            if (href && href.includes('/series/') && !href.includes('/chapter')) {
                                uniqueLinks.add(href);
                            }
                        });
                    });
                    
                    return Array.from(uniqueLinks);
                }''')
                
                old_count = len(all_links)
                all_links.update(links)
                new_count = len(all_links) - old_count
                
                print(f"   ✓ Found {new_count} new links | Total: {len(all_links)}")
                
                # If no new links found on this page, might be past the last page
                if new_count == 0:
                    print(f"   ⚠ No new links found, reached end of catalog")
                    break
                
                # Check if we've reached target
                if len(all_links) >= max_links:
                    print(f"\n✅ Target reached: {len(all_links)} links collected")
                    break
                
                # Move to next page
                current_page += 1
                await asyncio.sleep(2)  # Be nice to the server
                
        except Exception as e:
            print(f"\n✗ Error during scraping: {str(e)}")
        
        finally:
            await browser.close()
    
    all_links = list(all_links)[:max_links]
    print(f"\n📊 Scraping finished!")
    print(f"   Pages visited: {current_page - start_page + 1}")
    print(f"   Unique links: {len(all_links)}")
    
    return all_links, current_page


def create_database(links, last_page):
    """Create database from scraped links"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"manga_database_{timestamp}.json"
    
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
    
    for idx, link in enumerate(links, 1):
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
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(manga_database, f, indent=2, ensure_ascii=False)
    
    return filename, manga_database


async def main(start_page, max_links):
    """Main function"""
    print("="*60)
    print("Asura Comics Manga Scraper")
    print("="*60 + "\n")
    
    # Scrape
    links, last_page = await scrape_asura_manga(start_page=start_page, max_links=max_links)
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total links collected: {len(links)}")
    print(f"Last page reached: {last_page}")
    
    # Create database
    filename, database = create_database(links, last_page)
    
    print(f"\n✓ Database saved to: {filename}")
    print(f"  - Total entries: {len(links)}")
    print(f"  - All marked as: Incomplete")
    
    # Sample entries
    print(f"\nSample database entries (first 10):")
    for entry in database["manga"][:10]:
        print(f"  {entry['id']}. {entry['title']}")
        print(f"     URL: {entry['url']}")
        print(f"     Status: {'✓ Complete' if entry['completed'] else '○ Pending'}")
        print()
    
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scrape manga links from Asura Comics'
    )
    
    parser.add_argument('--start-page', type=int, default=1,
                       help='Starting page number (default: 1)')
    parser.add_argument('--max-links', type=int, default=1000,
                       help='Maximum number of links to collect (default: 1000)')
    
    args = parser.parse_args()
    
    # Run scraper
    filename = asyncio.run(main(args.start_page, args.max_links))
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"Database: {filename}")
    print("="*60)