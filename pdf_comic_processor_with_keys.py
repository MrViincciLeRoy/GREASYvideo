
# pdf_comic_processor_with_keys.py
"""
PDF Comic Processor with Enhanced API Key Management
"""

import os
from pathlib import Path
import fitz
from typing import List, Dict
import json
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

# Import enhanced key manager
from enhanced_api_key_manager import APIKeyManager, ManagedGroqClient
from character_tracker import CharacterTracker, CharacterGuide
from analyzer_with_key_manager import CharacterAwareComicAnalyzer, PageAnalysis

@dataclass
class ComicBookAnalysis:
    """Complete comic book analysis data with character tracking."""
    title: str
    total_pages: int
    pages: List[PageAnalysis]
    output_directory: str
    character_guide: Dict = None

    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'total_pages': self.total_pages,
            'pages': [p.to_dict() for p in self.pages],
            'output_directory': self.output_directory,
            'character_guide': self.character_guide,
            'summary': {
                'total_panels': sum(p.total_panels for p in self.pages),
                'total_narration_boxes': sum(len(p.narration_boxes) for p in self.pages),
                'successful_pages': sum(1 for p in self.pages if p.success),
                'failed_pages': sum(1 for p in self.pages if not p.success),
                'unique_characters_tracked': len(set(
                    char_id
                    for page in self.pages
                    for char_id in (page.characters_on_page or [])
                ))
            }
        }

class PDFComicProcessor:
    """Process PDF comics with API key rotation and character tracking"""

    def __init__(self, key_manager: APIKeyManager, character_guide: CharacterGuide,
                 comic_format: str = "auto"):
        """
        Initialize with key manager and character guide

        Args:
            key_manager: APIKeyManager instance with configured keys
            character_guide: CharacterGuide for character tracking
            comic_format: "auto", "traditional", or "webtoon"
        """
        self.key_manager = key_manager
        self.character_guide = character_guide
        self.comic_format = comic_format

        # Initialize analyzer with key manager
        self.analyzer = CharacterAwareComicAnalyzer(
            key_manager=key_manager,
            character_guide=character_guide,
            comic_format=comic_format
        )

        print("✓ PDF Comic Processor initialized")
        print(f"  Protagonist: {character_guide.protagonist_name or 'Unknown'}")
        print(f"  Comic format: {comic_format}")
        print(f"  API keys: {len(key_manager.groq_keys)} Groq + "
              f"{'1 HF' if key_manager.huggingface_token else '0 HF'}")

    def extract_pages_from_pdf(self, pdf_path: str, output_dir: str = "extracted_pages",
                               start_page: int = 1, end_page: int = None,
                               dpi: int = 150) -> List[str]:
        """Extract pages from PDF as images"""
        print(f"\n📄 Opening PDF: {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"✓ PDF loaded: {total_pages} pages")

        start_idx = start_page - 1
        end_idx = end_page if end_page else total_pages
        end_idx = min(end_idx, total_pages)

        print(f"📖 Extracting pages {start_page} to {end_idx}...")

        page_paths = []
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        for page_num in tqdm(range(start_idx, end_idx), desc="Extracting pages"):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)

            page_filename = f"page_{page_num + 1:03d}.png"
            page_path = os.path.join(output_dir, page_filename)
            pix.save(page_path)
            page_paths.append(page_path)

        doc.close()
        print(f"✅ Extracted {len(page_paths)} pages to {output_dir}")

        return page_paths

    def process_pdf_comic(self, pdf_path: str, output_dir: str = "comic_analysis",
                         start_page: int = 1, end_page: int = None,
                         extract_dpi: int = 150) -> ComicBookAnalysis:
        """Complete pipeline with API key rotation"""
        print("\n" + "="*80)
        print("🎨 PDF COMIC ANALYZER - WITH API KEY ROTATION")
        print("="*80)

        # Show initial API status
        self.key_manager.print_status()

        os.makedirs(output_dir, exist_ok=True)
        pages_dir = os.path.join(output_dir, "extracted_pages")

        comic_title = Path(pdf_path).stem
        print(f"📚 Comic: {comic_title}")

        # Extract pages
        page_paths = self.extract_pages_from_pdf(
            pdf_path=pdf_path,
            output_dir=pages_dir,
            start_page=start_page,
            end_page=end_page,
            dpi=extract_dpi
        )

        # Process each page
        print(f"\n🤖 Analyzing {len(page_paths)} pages with character tracking...")
        print("="*80 + "\n")

        page_analyses = []

        for i, page_path in enumerate(page_paths, start=start_page):
            print(f"\n{'─'*80}")
            print(f"Processing Page {i}/{len(page_paths) + start_page - 1}")
            print(f"{'─'*80}")

            # Show API status every 5 pages
            if i % 5 == 0:
                self.key_manager.print_status()

            try:
                page_analysis = self.analyzer.analyze_complete_page(
                    page_image_path=page_path,
                    page_number=i,
                    output_dir=output_dir
                )
                page_analyses.append(page_analysis)

                self.analyzer.save_analysis_with_characters(
                    page_analysis,
                    output_dir=output_dir
                )

            except Exception as e:
                print(f"❌ Error processing page {i}: {e}")
                failed_page = PageAnalysis(
                    page_number=i,
                    page_context=f"Error: {str(e)}",
                    total_panels=0,
                    panels=[],
                    narration_boxes=[],
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    characters_on_page=[]
                )
                page_analyses.append(failed_page)

        # Save character tracking
        print(f"\n{'='*80}")
        print("💾 Saving complete character tracking...")
        print(f"{'='*80}")

        self.analyzer.save_complete_character_tracking(output_dir)

        # Create complete analysis
        comic_analysis = ComicBookAnalysis(
            title=comic_title,
            total_pages=len(page_analyses),
            pages=page_analyses,
            output_directory=output_dir,
            character_guide=self.character_guide.to_dict()
        )

        # Save complete analysis
        self._save_complete_analysis(comic_analysis, output_dir)

        # Final status
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)

        # Show final API usage
        print("\n📊 Final API Usage:")
        self.key_manager.print_status()

        summary = comic_analysis.to_dict()['summary']
        print(f"\n📈 Results:")
        print(f"   • Total pages: {comic_analysis.total_pages}")
        print(f"   • Total panels: {summary['total_panels']}")
        print(f"   • Successful: {summary['successful_pages']} pages")
        print(f"   • Characters tracked: {summary['unique_characters_tracked']}")

        return comic_analysis

    def _save_complete_analysis(self, comic_analysis: ComicBookAnalysis, output_dir: str):
        """Save complete analysis"""
        master_json_path = os.path.join(output_dir, "complete_analysis.json")
        with open(master_json_path, 'w', encoding='utf-8') as f:
            json.dump(comic_analysis.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved: {master_json_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     PDF COMIC PROCESSOR WITH API KEY ROTATION                ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    PDF_PATH = "/content/drive/MyDrive/[CH - 1] The Knight King Who Returned with a God.pdf"
    OUTPUT_DIR = "knight_king_analysis"
    START_PAGE = 2
    END_PAGE = None
    DPI = 150
    COMIC_FORMAT = "auto"

    # API KEYS - Add your keys here
    GROQ_KEYS = [
        os.getenv('GROQ_KEY'), 
        os.getenv('GROQ_KEY_2'), 
       os.getenv('GROQ_KEY_3'), 
       os.getenv('GROQ_KEY_4')
        # Add more keys here for higher capacity
    ]

    HUGGINGFACE_TOKEN = None  # Optional fallback

    # CHARACTER SETUP
    character_guide = CharacterGuide(
        protagonist_name=None,
        protagonist_gender="male",
        protagonist_role="Young Knight",
        protagonist_description="Young boy, noble birth",
        protagonist_visual_cues=["blonde hair", "young boy", "blue eyes"],
        known_characters=[
            {'name': 'Nanny', 'gender': 'female', 'role': 'Caretaker'}
        ],
        setting="Fantasy kingdom",
        tone="Epic",
        story_theme="Rebirth"
    )

    # ========================================================================
    # RUN
    # ========================================================================

    # Create key manager
    key_manager = APIKeyManager(
        groq_keys=GROQ_KEYS,
        huggingface_token=HUGGINGFACE_TOKEN,
        state_file="api_keys.json"
    )

    # Create processor
    processor = PDFComicProcessor(
        key_manager=key_manager,
        character_guide=character_guide,
        comic_format=COMIC_FORMAT
    )

    # Process
    try:
        comic_analysis = processor.process_pdf_comic(
            pdf_path=PDF_PATH,
            output_dir=OUTPUT_DIR,
            start_page=START_PAGE,
            end_page=END_PAGE,
            extract_dpi=DPI
        )

        print("\n🎉 Processing complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()