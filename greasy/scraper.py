#!/usr/bin/env python3
"""
AsuraComics chapter scraper - worker side.
Scrapes page image URLs for a given chapter range and downloads them as images.
No PDFs stored — images downloaded, used, then deleted.
"""

import asyncio
import os
import re
import time
import requests
from pathlib import Path
from typing import List, Tuple
from playwright.async_api import async_playwright


BASE = "https://asuracomic.net"


async def get_chapter_list(series_url: str) -> List[Tuple[int, str]]:
    """
    Returns [(chapter_number, chapter_url), ...] sorted ascending.
    Works by scraping the series page for all chapter links.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await ctx.new_page()

        url = series_url if series_url.startswith("http") else f"https://{series_url}"
        print(f"  Fetching chapter list: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        chapters = await page.evaluate("""() => {
            const results = [];
            const links = document.querySelectorAll('a[href*="/chapter/"]');
            links.forEach(a => {
                const href = a.href;
                const match = href.match(/chapter[/-](\\d+(?:\\.\\d+)?)/i);
                if (match) {
                    results.push({ num: parseFloat(match[1]), url: href });
                }
            });
            return results;
        }""")

        await browser.close()

    seen = {}
    for ch in chapters:
        n = ch["num"]
        if n not in seen:
            seen[n] = ch["url"]

    sorted_chapters = sorted(seen.items())
    print(f"  Found {len(sorted_chapters)} chapters total")
    return sorted_chapters


async def get_chapter_page_urls(chapter_url: str) -> List[str]:
    """
    Returns list of image URLs for all pages in a chapter.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = await ctx.new_page()

        await page.goto(chapter_url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(4)

        # Scroll to trigger lazy loading
        await page.evaluate("""async () => {
            for (let i = 0; i < 10; i++) {
                window.scrollBy(0, window.innerHeight);
                await new Promise(r => setTimeout(r, 500));
            }
        }""")
        await asyncio.sleep(2)

        imgs = await page.evaluate("""() => {
            const urls = [];
            const selectors = [
                'img[src*="asuracomic"]',
                'img[src*="cdn"]',
                '.chapter-content img',
                '.reading-content img',
                'img[class*="chapter"]',
                'img[class*="page"]'
            ];
            const seen = new Set();
            selectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(img => {
                    const src = img.src || img.dataset.src || img.dataset.lazy;
                    if (src && !seen.has(src) && /\\.(jpg|jpeg|png|webp)/i.test(src)) {
                        // Skip small UI images (icons, logos)
                        if (img.naturalWidth > 200 || img.width > 200) {
                            seen.add(src);
                            urls.push(src);
                        }
                    }
                });
            });
            return urls;
        }""")

        await browser.close()

    return imgs


def download_image(url: str, dest: str, retries: int = 3) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": BASE
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=20, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"    ✗ Failed {url}: {e}")
    return False


async def scrape_chapter_range(
    series_url: str,
    start_chapter: int,
    end_chapter: int,
    output_dir: str
) -> List[str]:
    """
    Scrapes chapters start_chapter..end_chapter (inclusive).
    Downloads all page images into output_dir/chapter_XXX/page_YYY.jpg
    Returns list of chapter directories (sorted).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SCRAPING: {series_url}")
    print(f"Chapters: {start_chapter} → {end_chapter}")
    print(f"{'='*60}")

    # Get full chapter list
    all_chapters = await get_chapter_list(series_url)

    # Filter to requested range
    target = [(n, u) for n, u in all_chapters if start_chapter <= n <= end_chapter]

    if not target:
        raise RuntimeError(
            f"No chapters found in range {start_chapter}-{end_chapter}. "
            f"Available: {[n for n, _ in all_chapters[:5]]}..."
        )

    print(f"  Chapters to scrape: {len(target)}")

    chapter_dirs = []

    for ch_num, ch_url in target:
        ch_dir = output_dir / f"chapter_{int(ch_num):03d}"
        ch_dir.mkdir(exist_ok=True)

        print(f"\n  📖 Chapter {ch_num}: {ch_url}")
        page_urls = await get_chapter_page_urls(ch_url)

        if not page_urls:
            print(f"    ⚠ No pages found, skipping")
            continue

        print(f"    Pages: {len(page_urls)}")

        downloaded = 0
        for i, img_url in enumerate(page_urls, 1):
            ext = "jpg"
            for fmt in ["png", "webp", "jpeg"]:
                if fmt in img_url.lower():
                    ext = fmt
                    break

            dest = str(ch_dir / f"page_{i:03d}.{ext}")
            if os.path.exists(dest):
                downloaded += 1
                continue

            ok = download_image(img_url, dest)
            if ok:
                downloaded += 1

        print(f"    ✓ Downloaded {downloaded}/{len(page_urls)} pages → {ch_dir}")
        chapter_dirs.append(str(ch_dir))

        # Be polite
        await asyncio.sleep(2)

    print(f"\n✓ Scrape complete — {len(chapter_dirs)} chapters ready")
    return sorted(chapter_dirs)


def cleanup_chapter_images(chapter_dirs: List[str]):
    """Delete downloaded images after processing to save disk space."""
    import shutil
    for d in chapter_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  🗑  Deleted {d}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape AsuraComics chapter range")
    parser.add_argument("series_url", help="AsuraComics series URL")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", default="scraped_chapters")
    args = parser.parse_args()

    dirs = asyncio.run(scrape_chapter_range(
        args.series_url, args.start, args.end, args.output
    ))
    print(f"\nChapter dirs: {dirs}")
