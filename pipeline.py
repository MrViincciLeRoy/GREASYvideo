#!/usr/bin/env python3
"""
Master Comic-to-Video Pipeline
Accepts either a PDF file or a directory of images (scraped chapter pages).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from greasy.core.keys import APIKeyManager
from greasy.core.tracker import CharacterGuide
from greasy.processing.pdf import PDFComicProcessor
from greasy.processing.analyzer import CharacterAwareComicAnalyzer, PageAnalysis
from greasy.story.generator import ComicStoryGenerator, StoryContext
from greasy.video.generator import KokoroTTSVideoGenerator


def images_from_dir(image_dir: str) -> list:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in exts
    ])


class ComicPipeline:

    def __init__(self, groq_keys: list, hf_token: str = None, comic_format: str = "auto"):
        self.groq_keys = [k for k in groq_keys if k]
        self.comic_format = comic_format

        self.key_manager = APIKeyManager(
            groq_keys=self.groq_keys,
            huggingface_token=hf_token or None,
            state_file="pipeline_api_keys.json"
        )

        hf_status = "✓ HuggingFace fallback ready" if hf_token else "✗ No HuggingFace token"
        print(f"✓ Pipeline ready — {len(self.groq_keys)} Groq keys, format: {comic_format}")
        print(f"  {hf_status}")

    def parse_input_name(self, input_path: str) -> tuple:
        import re
        name = Path(input_path).stem
        series = re.sub(r'[^\w\s-]', '', name)
        series = re.sub(r'[-\s]+', '_', series)
        series = '_'.join(w.capitalize() for w in series.split('_'))
        return series, "output"

    def run(self, input_path: str, character_config: dict,
            start_page: int = 1, end_page: int = None,
            extract_dpi: int = 150, video_voice: str = "af_bella",
            video_batch_size: int = 3, video_resolution: tuple = (720, 1280),
            base_output_dir: str = "comic_output"):

        input_path = Path(input_path)
        is_dir = input_path.is_dir()
        is_pdf = input_path.suffix.lower() == ".pdf"

        if not is_dir and not is_pdf:
            print(f"❌ Input must be a PDF file or image directory: {input_path}")
            sys.exit(1)

        series_name, identifier = self.parse_input_name(str(input_path))
        output_base = Path(base_output_dir) / series_name / identifier
        paths = {k: str(output_base / k) for k in ["extracted_pages", "analysis", "story", "video"]}
        paths["base"] = str(output_base)
        for p in paths.values():
            os.makedirs(p, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Input  : {input_path}")
        print(f"Type   : {'Directory' if is_dir else 'PDF'}")
        print(f"Output : {output_base}")
        print(f"{'='*70}")

        character_guide = CharacterGuide(
            protagonist_name=character_config.get("protagonist_name"),
            protagonist_gender=character_config.get("protagonist_gender", "unspecified"),
            protagonist_role=character_config.get("protagonist_role", "protagonist"),
            protagonist_description=character_config.get("protagonist_description", ""),
            protagonist_visual_cues=character_config.get("protagonist_visual_cues", []),
            known_characters=character_config.get("known_characters", []),
            setting=character_config.get("setting", ""),
            tone=character_config.get("tone", ""),
            story_theme=character_config.get("story_theme", "")
        )

        # STEP 1: Page extraction / analysis
        print(f"\n{'='*70}")
        print("STEP 1/3: PAGE EXTRACTION")
        print(f"{'='*70}")

        if is_pdf:
            processor = PDFComicProcessor(
                key_manager=self.key_manager,
                character_guide=character_guide,
                comic_format=self.comic_format
            )
            comic_analysis = processor.process_pdf_comic(
                pdf_path=str(input_path),
                output_dir=paths["analysis"],
                start_page=start_page,
                end_page=end_page,
                extract_dpi=extract_dpi
            )
        else:
            page_images = images_from_dir(str(input_path))
            if not page_images:
                print(f"❌ No images found in {input_path}")
                sys.exit(1)

            print(f"✓ Found {len(page_images)} images in directory")

            analyzer = CharacterAwareComicAnalyzer(
                key_manager=self.key_manager,
                character_guide=character_guide,
                comic_format=self.comic_format
            )

            page_analyses = []
            for i, img_path in enumerate(page_images, 1):
                print(f"\n  Page {i}/{len(page_images)}: {Path(img_path).name}")
                pa = analyzer.analyze_complete_page(
                    page_image_path=img_path,
                    page_number=i,
                    output_dir=paths["analysis"]
                )
                page_analyses.append(pa)
                analyzer.save_analysis_with_characters(pa, output_dir=paths["analysis"])

            analyzer.save_complete_character_tracking(paths["analysis"])

            from greasy.processing.pdf import ComicBookAnalysis
            comic_analysis = ComicBookAnalysis(
                title=series_name,
                total_pages=len(page_analyses),
                pages=page_analyses,
                output_directory=paths["analysis"],
                character_guide=character_guide.to_dict()
            )

            analysis_json = os.path.join(paths["analysis"], "complete_analysis.json")
            with open(analysis_json, "w", encoding="utf-8") as f:
                json.dump(comic_analysis.to_dict(), f, indent=2, ensure_ascii=False)

        analysis_json_path = os.path.join(paths["analysis"], "complete_analysis.json")

        # STEP 2: Story generation
        print(f"\n{'='*70}")
        print("STEP 2/3: STORY GENERATION")
        print(f"{'='*70}")

        story_context = StoryContext(
            main_hero=character_config.get("protagonist_name", "the protagonist"),
            hero_role=character_config.get("protagonist_role", "hero"),
            hero_gender=character_config.get("protagonist_gender", "male"),
            setting=character_config.get("setting", ""),
            tone=character_config.get("tone", "adventure"),
            additional_characters=character_config.get("known_characters", []),
            story_theme=character_config.get("story_theme", "")
        )

        story_gen = ComicStoryGenerator(self.key_manager)
        story_gen.generate(
            analysis_path=analysis_json_path,
            context=story_context,
            output_dir=paths["story"]
        )

        story_mapping = os.path.join(paths["story"], "panel_to_story_mapping.json")

        # STEP 3: Video
        print(f"\n{'='*70}")
        print("STEP 3/3: VIDEO GENERATION")
        print(f"{'='*70}")

        video_file = f"{series_name}_{identifier}_video.mp4"
        video_out = os.path.join(paths["video"], video_file)

        gen = KokoroTTSVideoGenerator(
            json_path=story_mapping,
            base_panels_folder=paths["analysis"],
            output_path=video_out,
            include_audio=True,
            batch_size=video_batch_size,
            voice=video_voice,
            lang_code="a"
        )
        try:
            gen.generate_video(fps=24, resolution=video_resolution)
        finally:
            gen.cleanup()

        print(f"\n{'='*70}")
        print(f"✅ DONE: {video_out}")
        print(f"{'='*70}")
        return video_out


def main():
    parser = argparse.ArgumentParser(
        description="Comic-to-Video Pipeline — PDF or image directory input"
    )
    parser.add_argument("input", help="PDF file or directory of page images")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", default="auto", choices=["auto", "traditional", "webtoon"])
    parser.add_argument("--voice", default="af_bella")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--output-dir", default="comic_output")
    args = parser.parse_args()

    groq_keys = [
        os.getenv("GROQ_KEY"),
        os.getenv("GROQ_KEY_2"),
        os.getenv("GROQ_KEY_3"),
        os.getenv("GROQ_KEY_4"),
    ]

    hf_token = os.getenv("HF_TOKEN")

    character_config = {
        "protagonist_name": None,
        "protagonist_gender": "male",
        "protagonist_role": "Hero",
        "protagonist_description": "Main character",
        "protagonist_visual_cues": ["young", "protagonist"],
        "known_characters": [],
        "setting": "Fantasy world",
        "tone": "Adventure",
        "story_theme": "Journey"
    }

    pipeline = ComicPipeline(groq_keys=groq_keys, hf_token=hf_token, comic_format=args.format)
    pipeline.run(
        input_path=args.input,
        character_config=character_config,
        start_page=args.start_page,
        end_page=args.end_page,
        extract_dpi=args.dpi,
        video_voice=args.voice,
        video_batch_size=args.batch_size,
        base_output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
