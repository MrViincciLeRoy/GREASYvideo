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
    Scrape manga links from Asura Comics
    """
    base_url = "https://asuracomic.net/series?page=1&genres=&status=-1&types=-1&order=bookmarks"
    all_links = set()
    current_page = 1
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        print(f"Starting scrape...")
        print(f"Target: {max_links} links\n")
        
        try:
            print("Loading initial page...")
            await page.goto(base_url, wait_until='domcontentloaded', timeout=60000)
            await asyncio.sleep(3)
            
            # Navigate to start_page if needed
            if start_page > 1:
                print(f"Navigating to page {start_page}...")
                for i in range(1, start_page):
                    next_btn = await page.query_selector('a[rel="next"]')
                    if not next_btn:
                        next_btn = await page.query_selector('a:has-text("Next")')
                    
                    if next_btn:
                        await next_btn.click()
                        await asyncio.sleep(2)
                        current_page += 1
                        print(f"  Navigated to page {current_page}")
                    else:
                        print(f"  Could not find next button at page {current_page}")
                        break
            
            print(f"\nStarting collection from page {current_page}...\n")
            
            while len(all_links) < max_links:
                print(f"Scraping page {current_page}...")
                
                # Wait for content to load - increased wait time
                await asyncio.sleep(5)
                
                # Wait for the grid/content to be visible
                try:
                    await page.wait_for_selector('.grid, .series-list, a[href*="/series/"]', timeout=10000)
                except:
                    print(f"  ⚠ Timeout waiting for content")
                
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
                
                print(f"  Found {new_count} new links (Total: {len(all_links)})")
                
                # If no new links found, wait a bit more and try again
                if new_count == 0 and len(all_links) > 0:
                    print(f"  ⚠ No new links, waiting 5s and retrying...")
                    await asyncio.sleep(5)
                    continue
                
                if len(all_links) >= max_links:
                    print(f"\n✓ Target reached: {len(all_links)} links collected")
                    break
                
                # Find next button
                next_button = await page.query_selector('a[rel="next"]')
                if not next_button:
                    next_button = await page.query_selector('a:has-text("Next")')
                if not next_button:
                    next_button = await page.query_selector('nav a:last-child')
                
                if not next_button:
                    print(f"  ⚠ Next button not found, waiting 3s and retrying...")
                    await asyncio.sleep(3)
                    next_button = await page.query_selector('a[rel="next"]')
                    if not next_button:
                        print("\n✗ No more pages available")
                        break
                
                is_disabled = await page.evaluate('''(btn) => {
                    return btn.classList.contains('disabled') || 
                           btn.hasAttribute('disabled') ||
                           btn.getAttribute('aria-disabled') === 'true';
                }''', next_button)
                
                if is_disabled:
                    print("\n✗ Reached last page")
                    break
                
                print(f"  Clicking next button...")
                
                # Scroll to button before clicking
                await page.evaluate('(btn) => btn.scrollIntoView()', next_button)
                await asyncio.sleep(1)
                
                # Click and wait for navigation
                await next_button.click()
                current_page += 1
                
                # Wait longer for page to load after click
                await asyncio.sleep(5)
                
                # Wait for URL to change or content to refresh
                try:
                    await page.wait_for_load_state('networkidle', timeout=10000)
                except:
                    print(f"  ⚠ Network not idle, continuing anyway...")
                
        except Exception as e:
            print(f"\n✗ Error during scraping: {str(e)}")
        
        finally:
            await browser.close()
    
    all_links = list(all_links)[:max_links]
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