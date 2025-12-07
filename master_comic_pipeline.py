#!/usr/bin/env python3
"""
Master Comic-to-Video Pipeline
Orchestrates: PDF Processing → Story Generation → Video Creation
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Import all necessary components
from enhanced_api_key_manager import APIKeyManager
from character_tracker import CharacterGuide
from pdf_comic_processor_with_keys import PDFComicProcessor
from story_generator_with_keys import ComicStoryGenerator, StoryContext
from VidGen import KokoroTTSVideoGenerator


class ComicPipeline:
    """Master pipeline orchestrator"""
    
    def __init__(self, groq_keys: list, comic_format: str = "auto"):
        """
        Initialize pipeline
        
        Args:
            groq_keys: List of Groq API keys
            comic_format: "auto", "traditional", or "webtoon"
        """
        self.groq_keys = groq_keys
        self.comic_format = comic_format
        
        # Initialize API key manager
        self.key_manager = APIKeyManager(
            groq_keys=groq_keys,
            huggingface_token=None,
            state_file="pipeline_api_keys.json"
        )
        
        print("✓ Pipeline initialized")
        print(f"  API Keys: {len(groq_keys)} Groq keys configured")
        print(f"  Comic Format: {comic_format}")
    
    def parse_comic_name(self, pdf_path: str) -> tuple:
        """
        Parse comic name from PDF filename
        
        Examples:
            "Chainsaw Man [Volume 01].pdf" → ("Chainsaw_Man", "Vol01")
            "[CH - 1] The Knight King Who Returned with a God.pdf" → ("Knight_King_Who_Returned_With_God", "CH1")
            "Demon Slayer Vol 1.pdf" → ("Demon_Slayer", "Vol1")
            "One Piece Chapter 1000.pdf" → ("One_Piece", "CH1000")
        """
        filename = Path(pdf_path).stem
        original_filename = filename
        
        print(f"\n📖 Parsing filename: '{filename}'")
        
        # Initialize
        series_name = None
        identifier = None
        
        # PATTERN 1: "[CH - X] Series Name" format
        # Example: "[CH - 1] The Knight King Who Returned with a God"
        if filename.startswith('[CH') or filename.startswith('[ch'):
            import re
            match = re.match(r'\[CH\s*-?\s*(\d+)\]\s*(.+)', filename, re.IGNORECASE)
            if match:
                chapter_num = match.group(1)
                series_name = match.group(2).strip()
                identifier = f"CH{int(chapter_num)}"
                print(f"   Pattern: [CH - X] format")
                print(f"   Series: {series_name}")
                print(f"   Chapter: {identifier}")
        
        # PATTERN 2: "Series Name [Volume XX]" format
        # Example: "Chainsaw Man [Volume 01]"
        elif '[volume' in filename.lower() or '[vol' in filename.lower():
            import re
            match = re.match(r'(.+?)\s*\[(volume|vol)\s*(\d+)\]', filename, re.IGNORECASE)
            if match:
                series_name = match.group(1).strip()
                vol_num = match.group(3)
                identifier = f"Vol{int(vol_num):02d}"
                print(f"   Pattern: Series [Volume XX] format")
                print(f"   Series: {series_name}")
                print(f"   Volume: {identifier}")
        
        # PATTERN 3: "Series Name Vol X" or "Series Name Volume X" format
        # Example: "Demon Slayer Vol 1"
        elif 'vol' in filename.lower() and not series_name:
            import re
            # Try "Vol X" or "Volume X" pattern
            match = re.match(r'(.+?)\s+(vol|volume)\s*(\d+)', filename, re.IGNORECASE)
            if match:
                series_name = match.group(1).strip()
                vol_num = match.group(3)
                identifier = f"Vol{int(vol_num)}"
                print(f"   Pattern: Series Vol X format")
                print(f"   Series: {series_name}")
                print(f"   Volume: {identifier}")
        
        # PATTERN 4: "Series Name Chapter X" or "Series Name Ch X" format
        # Example: "One Piece Chapter 1000"
        elif ('chapter' in filename.lower() or ' ch ' in filename.lower()) and not series_name:
            import re
            match = re.match(r'(.+?)\s+(chapter|ch)\s*(\d+)', filename, re.IGNORECASE)
            if match:
                series_name = match.group(1).strip()
                ch_num = match.group(3)
                identifier = f"CH{int(ch_num)}"
                print(f"   Pattern: Series Chapter X format")
                print(f"   Series: {series_name}")
                print(f"   Chapter: {identifier}")
        
        # FALLBACK: Use full filename as series name
        if not series_name:
            series_name = filename
            identifier = "Vol1"
            print(f"   Pattern: No volume/chapter info detected")
            print(f"   Using full name as series: {series_name}")
        
        # Clean up series name
        # Remove common bracketed info and extra spaces
        series_name = series_name.replace('[', '').replace(']', '')
        series_name = series_name.strip()
        
        # Remove common words that might be left over
        cleanup_words = ['volume', 'vol', 'chapter', 'ch']
        words = series_name.split()
        cleaned_words = []
        for word in words:
            if word.lower() not in cleanup_words:
                cleaned_words.append(word)
        
        if cleaned_words:
            series_name = ' '.join(cleaned_words)
        
        # Convert to filesystem-safe name
        # Capitalize each word and join with underscores
        series_name = '_'.join(word.capitalize() for word in series_name.split())
        
        # Remove any remaining special characters
        import re
        series_name = re.sub(r'[^\w\s-]', '', series_name)
        series_name = re.sub(r'[-\s]+', '_', series_name)
        
        print(f"   ✓ Final: {series_name}/{identifier}")
        
        return series_name, identifier
    
    def create_folder_structure(self, base_dir: str, series_name: str, 
                               identifier: str) -> dict:
        """
        Create consistent folder structure
        
        Returns dict with all paths
        """
        # Main comic folder: Demon_Slayer/Vol1/
        comic_dir = Path(base_dir) / series_name / identifier
        
        paths = {
            'base': str(comic_dir),
            'extracted_pages': str(comic_dir / 'extracted_pages'),
            'analysis': str(comic_dir / 'analysis'),
            'story': str(comic_dir / 'story'),
            'video': str(comic_dir / 'video'),
            'temp': str(comic_dir / 'temp')
        }
        
        # Create all directories
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        
        print(f"\n📁 Folder structure created:")
        print(f"   Base: {paths['base']}")
        for key, path in paths.items():
            if key != 'base':
                print(f"   └── {key}: {Path(path).name}/")
        
        return paths
    
    def run_full_pipeline(self, pdf_path: str, 
                          character_config: dict,
                          start_page: int = 1,
                          end_page: int = None,
                          extract_dpi: int = 150,
                          video_voice: str = "af_bella",
                          video_batch_size: int = 3,
                          video_resolution: tuple = (1080, 1920),
                          base_output_dir: str = "comic_output"):
        """
        Run complete pipeline: PDF → Analysis → Story → Video
        
        Args:
            pdf_path: Path to PDF file
            character_config: Dict with character setup
            start_page: First page to process (1-indexed)
            end_page: Last page to process (None = all)
            extract_dpi: DPI for page extraction
            video_voice: Kokoro TTS voice
            video_batch_size: Video generation batch size
            video_resolution: Video resolution (width, height)
            base_output_dir: Base directory for all outputs
        """
        print("\n" + "="*80)
        print("🚀 STARTING FULL COMIC-TO-VIDEO PIPELINE")
        print("="*80)
        
        # Parse comic name and create folders
        series_name, identifier = self.parse_comic_name(pdf_path)
        print(f"\n📚 Comic: {series_name} - {identifier}")
        
        paths = self.create_folder_structure(base_output_dir, series_name, identifier)
        
        # Create character guide
        character_guide = CharacterGuide(
            protagonist_name=character_config.get('protagonist_name'),
            protagonist_gender=character_config.get('protagonist_gender', 'unspecified'),
            protagonist_role=character_config.get('protagonist_role', 'protagonist'),
            protagonist_description=character_config.get('protagonist_description', ''),
            protagonist_visual_cues=character_config.get('protagonist_visual_cues', []),
            known_characters=character_config.get('known_characters', []),
            setting=character_config.get('setting', ''),
            tone=character_config.get('tone', ''),
            story_theme=character_config.get('story_theme', '')
        )
        
        # STEP 1: PDF PROCESSING
        print(f"\n{'='*80}")
        print("STEP 1/3: PDF PROCESSING & PANEL ANALYSIS")
        print(f"{'='*80}")
        
        processor = PDFComicProcessor(
            key_manager=self.key_manager,
            character_guide=character_guide,
            comic_format=self.comic_format
        )
        
        comic_analysis = processor.process_pdf_comic(
            pdf_path=pdf_path,
            output_dir=paths['analysis'],
            start_page=start_page,
            end_page=end_page,
            extract_dpi=extract_dpi
        )
        
        analysis_json_path = os.path.join(paths['analysis'], "complete_analysis.json")
        
        # STEP 2: STORY GENERATION
        print(f"\n{'='*80}")
        print("STEP 2/3: STORY GENERATION")
        print(f"{'='*80}")
        
        story_context = StoryContext(
            main_hero=character_config.get('protagonist_name', 'the protagonist'),
            hero_role=character_config.get('protagonist_role', 'hero'),
            hero_gender=character_config.get('protagonist_gender', 'male'),
            setting=character_config.get('setting', ''),
            tone=character_config.get('tone', 'adventure'),
            additional_characters=character_config.get('known_characters', []),
            story_theme=character_config.get('story_theme', '')
        )
        
        story_generator = ComicStoryGenerator(self.key_manager)
        
        story_output = story_generator.generate(
            analysis_path=analysis_json_path,
            context=story_context,
            output_dir=paths['story']
        )
        
        story_mapping_path = os.path.join(paths['story'], "panel_to_story_mapping.json")
        
        # STEP 3: VIDEO GENERATION
        print(f"\n{'='*80}")
        print("STEP 3/3: VIDEO GENERATION WITH TTS")
        print(f"{'='*80}")
        
        video_filename = f"{series_name}_{identifier}_video.mp4"
        video_output_path = os.path.join(paths['video'], video_filename)
        
        # Find base panels folder (where page_XXX_panels folders are)
        base_panels_folder = paths['analysis']
        
        video_generator = KokoroTTSVideoGenerator(
            json_path=story_mapping_path,
            base_panels_folder=base_panels_folder,
            output_path=video_output_path,
            include_audio=True,
            batch_size=video_batch_size,
            voice=video_voice,
            lang_code='a'
        )
        
        try:
            video_generator.generate_video(
                fps=24,
                resolution=video_resolution
            )
        finally:
            video_generator.cleanup()
        
        # FINAL SUMMARY
        print(f"\n{'='*80}")
        print("✅ PIPELINE COMPLETE!")
        print(f"{'='*80}")
        
        print(f"\n📊 Summary:")
        print(f"   Comic: {series_name} - {identifier}")
        print(f"   Pages processed: {comic_analysis.total_pages}")
        
        summary = comic_analysis.to_dict()['summary']
        print(f"   Total panels: {summary['total_panels']}")
        print(f"   Story segments: {story_output['total_segments']}")
        print(f"   Characters tracked: {summary['unique_characters_tracked']}")
        
        print(f"\n📁 Output Locations:")
        print(f"   Analysis: {paths['analysis']}")
        print(f"   Story: {paths['story']}")
        print(f"   Video: {video_output_path}")
        
        print(f"\n🎬 Final Video: {video_filename}")
        print(f"   Location: {video_output_path}")
        print(f"   Resolution: {video_resolution[0]}x{video_resolution[1]}")
        print(f"   Voice: {video_voice}")
        
        # Final API usage
        print(f"\n📊 Final API Usage:")
        self.key_manager.print_status()
        
        return {
            'series_name': series_name,
            'identifier': identifier,
            'paths': paths,
            'video_path': video_output_path,
            'analysis': comic_analysis.to_dict(),
            'story': story_output
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Comic-to-Video Pipeline: Convert PDF comics to narrated videos'
    )
    
    parser.add_argument('pdf_path', type=str, help='Path to PDF file')
    parser.add_argument('--start-page', type=int, default=1, help='First page (default: 1)')
    parser.add_argument('--end-page', type=int, default=None, help='Last page (default: all)')
    parser.add_argument('--dpi', type=int, default=150, help='Extraction DPI (default: 150)')
    parser.add_argument('--format', type=str, default='auto', 
                       choices=['auto', 'traditional', 'webtoon'],
                       help='Comic format (default: auto)')
    parser.add_argument('--voice', type=str, default='af_bella',
                       help='TTS voice (default: af_bella)')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Video batch size (default: 3)')
    parser.add_argument('--output-dir', type=str, default='comic_output',
                       help='Output directory (default: comic_output)')
    
    args = parser.parse_args()
    
    # Get API keys from environment
    groq_keys = [
        os.getenv('GROQ_KEY'),
        os.getenv('GROQ_KEY_2'),
        os.getenv('GROQ_KEY_3'),
        os.getenv('GROQ_KEY_4')
    ]
    groq_keys = [k for k in groq_keys if k]  # Filter None values
    
    if not groq_keys:
        print("❌ Error: No GROQ API keys found in environment!")
        print("   Set GROQ_KEY, GROQ_KEY_2, etc. environment variables")
        sys.exit(1)
    
    # Character configuration
    # TODO: Make this configurable via JSON file or arguments
    character_config = {
        'protagonist_name': None,  # Will be discovered
        'protagonist_gender': 'male',
        'protagonist_role': 'Hero',
        'protagonist_description': 'Main character',
        'protagonist_visual_cues': ['young', 'protagonist'],
        'known_characters': [],
        'setting': 'Fantasy world',
        'tone': 'Adventure',
        'story_theme': 'Journey'
    }
    
    # Initialize and run pipeline
    pipeline = ComicPipeline(
        groq_keys=groq_keys,
        comic_format=args.format
    )
    
    result = pipeline.run_full_pipeline(
        pdf_path=args.pdf_path,
        character_config=character_config,
        start_page=args.start_page,
        end_page=args.end_page,
        extract_dpi=args.dpi,
        video_voice=args.voice,
        video_batch_size=args.batch_size,
        base_output_dir=args.output_dir
    )
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    # Can also be used as a module
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive usage example
        print("""
╔══════════════════════════════════════════════════════════════╗
║           MASTER COMIC-TO-VIDEO PIPELINE                     ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python master_comic_pipeline.py <pdf_path> [options]

Example:
    python master_comic_pipeline.py "Demon Slayer Vol 1.pdf" --start-page 1 --end-page 50

Options:
    --start-page    First page to process (default: 1)
    --end-page      Last page to process (default: all)
    --dpi           Extraction DPI (default: 150)
    --format        Comic format: auto/traditional/webtoon (default: auto)
    --voice         TTS voice (default: af_bella)
    --batch-size    Video batch size (default: 3)
    --output-dir    Output directory (default: comic_output)

Environment Variables Required:
    GROQ_KEY, GROQ_KEY_2, GROQ_KEY_3, GROQ_KEY_4
    (At least one GROQ_KEY must be set)

For programmatic usage:
    from master_comic_pipeline import ComicPipeline
    
    pipeline = ComicPipeline(groq_keys=['key1', 'key2'])
    result = pipeline.run_full_pipeline(
        pdf_path='comic.pdf',
        character_config={...}
    )
""")