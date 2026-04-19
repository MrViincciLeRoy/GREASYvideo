import os
import base64
from PIL import Image
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from greasy.core.keys import APIKeyManager, ManagedGroqClient
from greasy.core.tracker import CharacterTracker, CharacterGuide
from greasy.processing.detector import PanelDetector


@dataclass
class PanelAnalysis:
    panel_id: int
    reading_order: int
    analysis: str
    ocr_text: str
    has_narration: bool
    position: Dict[str, int]
    dimensions: Dict[str, int]
    image_path: str
    success: bool
    characters_present: List[str] = None
    character_descriptions: Dict[str, str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'panel_id': self.panel_id,
            'reading_order': self.reading_order,
            'analysis': self.analysis,
            'ocr_text': self.ocr_text,
            'has_narration': self.has_narration,
            'position': self.position,
            'dimensions': self.dimensions,
            'image_path': self.image_path,
            'characters_present': self.characters_present or [],
            'character_descriptions': self.character_descriptions or {},
            'success': self.success,
            'error': self.error
        }


@dataclass
class PageAnalysis:
    page_number: int
    page_context: str
    total_panels: int
    panels: List[PanelAnalysis]
    narration_boxes: List[Dict]
    timestamp: str
    success: bool
    characters_on_page: List[str] = None

    def to_dict(self) -> Dict:
        return {
            'page_number': self.page_number,
            'page_context': self.page_context,
            'total_panels': self.total_panels,
            'panels': [p.to_dict() for p in self.panels],
            'narration_boxes': self.narration_boxes,
            'characters_on_page': self.characters_on_page or [],
            'timestamp': self.timestamp,
            'success': self.success,
            'metadata': {
                'successful_panels': sum(1 for p in self.panels if p.success),
                'failed_panels': sum(1 for p in self.panels if not p.success),
                'panels_with_narration': sum(1 for p in self.panels if p.has_narration),
                'total_narration_boxes': len(self.narration_boxes),
                'unique_characters': len(set(self.characters_on_page or []))
            }
        }


class CharacterAwareComicAnalyzer:

    def __init__(self, key_manager: APIKeyManager, character_guide: CharacterGuide,
                 comic_format: str = "auto"):
        self.key_manager = key_manager
        self.client = ManagedGroqClient(key_manager)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.character_tracker = CharacterTracker(character_guide)
        self.detector = PanelDetector(comic_format=comic_format)

        print(f"✓ Character-Aware Comic Analyzer initialized")
        print(f"✓ Tracking protagonist: {character_guide.protagonist_name or 'Unknown'}")
        print(f"✓ Known characters: {len(character_guide.known_characters)}")
        print(f"✓ Comic format: {comic_format}")
        print(f"✓ API keys: {len(key_manager.groq_keys)} Groq + "
              f"{'1 HuggingFace' if key_manager.huggingface_token else '0 HuggingFace'}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _resize_image_if_needed(self, image_path: str, max_size: int = 1024) -> str:
        img = Image.open(image_path)
        if max(img.size) <= max_size:
            return image_path

        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Fix: convert RGBA/P/LA modes before saving as JPEG
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # Always use .jpg extension for the resized temp file
        base = os.path.splitext(image_path)[0]
        temp_path = base + '_resized.jpg'
        img.save(temp_path, "JPEG", quality=85)
        return temp_path

    def analyze_panel_with_character_tracking(self, panel_path: str, panel_id: int,
                                               reading_order: int, page_number: int,
                                               context: str = "") -> Dict:
        print(f"  🖼️  Analyzing panel {panel_id} with character tracking...")

        if panel_id % 10 == 0:
            summary = self.key_manager.get_status_summary()
            current_key = summary['groq_keys'][self.key_manager.current_key_index % len(summary['groq_keys'])]
            print(f"      📊 API Status: {current_key['requests_remaining']} remaining")

        try:
            processed = self._resize_image_if_needed(panel_path)
            b64 = self._encode_image(processed)

            char_context = self.character_tracker.generate_prompt_context()

            prompt = f"""You are analyzing Panel #{reading_order} from a comic.

{char_context}

PAGE CONTEXT:
{context}

⚠️ CRITICAL CHARACTER IDENTIFICATION RULES:
1. If you see a character matching these visual cues: {', '.join(self.character_tracker.guide.protagonist_visual_cues)},
   THIS IS {self.character_tracker.guide.protagonist_name or 'THE PROTAGONIST'}
2. DO NOT describe them as "young boy", "the girl", "a child" - use their ACTUAL NAME
3. The protagonist is {self.character_tracker.guide.protagonist_gender}, use {self.character_tracker.guide.protagonist_gender} pronouns
4. Any character you see should be checked against the CHARACTER GUIDE above
5. Start your CHARACTERS section with: "PROTAGONIST ({self.character_tracker.guide.protagonist_name or 'Name Unknown'}): [description]"

YOUR ANALYSIS MUST INCLUDE:

1. CHARACTERS IDENTIFIED:
   - PROTAGONIST ({self.character_tracker.guide.protagonist_name or 'Name Unknown'}): [Is this character present? If yes, describe appearance, expression, what they're doing]
   - OTHER CHARACTERS: [List any other characters with their correct names from the guide, or "Unknown Character X" if new]

2. DIALOGUE & TEXT:
   - Transcribe ALL speech bubbles, narration boxes, sound effects
   - Attribute dialogue to specific characters by name

3. SCENE & ACTION:
   - What is happening in this moment
   - Setting and environment details

4. VISUAL ELEMENTS:
   - Art style, framing, colors, mood

5. STORY SIGNIFICANCE:
   - How this advances the narrative

Format your response clearly with these headers."""

            resp = self.client.chat_completions_create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                model=self.model,
                max_tokens=2000,
                temperature=0.7
            )

            analysis_text = resp.choices[0].message.content

            characters_found = self._extract_characters_from_analysis(
                analysis_text, page_number, panel_id, reading_order
            )

            result = {
                'panel_id': panel_id,
                'reading_order': reading_order,
                'analysis': analysis_text,
                'characters_present': characters_found,
                'success': True,
                'error': None
            }

            if processed != panel_path:
                os.remove(processed)

            return result

        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'panel_id': panel_id,
                'reading_order': reading_order,
                'analysis': "",
                'characters_present': [],
                'success': False,
                'error': str(e)
            }

    def _extract_characters_from_analysis(self, analysis: str, page: int,
                                           panel: int, reading_order: int) -> List[str]:
        characters_found = []
        analysis_lower = analysis.lower()

        protagonist = self.character_tracker.get_protagonist()
        if protagonist:
            visual_match = any(
                cue.lower() in analysis_lower
                for cue in self.character_tracker.guide.protagonist_visual_cues
            )

            if not visual_match:
                if protagonist.gender == "male":
                    male_terms = ["young boy", "the boy", "young man", "the protagonist",
                                  "main character", "he appears", "he is", "he has",
                                  "his hair", "his eyes", "his expression"]
                    visual_match = any(t in analysis_lower for t in male_terms)
                elif protagonist.gender == "female":
                    female_terms = ["young girl", "the girl", "young woman", "the protagonist",
                                    "main character", "she appears", "she is", "she has",
                                    "her hair", "her eyes", "her expression"]
                    visual_match = any(t in analysis_lower for t in female_terms)

            if visual_match:
                self.character_tracker.track_character_appearance(
                    character_id=protagonist.character_id,
                    page=page, panel=panel, reading_order=reading_order,
                    description=analysis[:200]
                )
                characters_found.append(protagonist.character_id)
                print(f"      ✓ Identified protagonist in panel {panel}")

        for char_id, character in self.character_tracker.characters.items():
            if char_id in characters_found:
                continue
            if character.name and character.name.lower() in analysis_lower:
                self.character_tracker.track_character_appearance(
                    character_id=char_id,
                    page=page, panel=panel, reading_order=reading_order,
                    description=analysis[:200]
                )
                characters_found.append(char_id)
                print(f"      ✓ Identified {character.name} in panel {panel}")

        return characters_found

    def analyze_complete_page(self, page_image_path: str, page_number: int,
                               output_dir: str = "output") -> PageAnalysis:
        print(f"\n{'='*80}")
        print(f"📖 Processing Page {page_number} (Character-Aware)")
        print(f"{'='*80}\n")

        self.key_manager.print_status()

        print(f"🔍 Analyzing full page context...")
        processed = self._resize_image_if_needed(page_image_path, max_size=1536)
        b64 = self._encode_image(processed)

        char_context = self.character_tracker.generate_prompt_context()

        prompt = f"""Analyze this comic page comprehensively:

{char_context}

Provide:
- Overall story beat and narrative
- Main characters present (use correct names from guide)
- Setting and emotional tone
- Key events on this page

Be detailed but concise."""

        try:
            resp = self.client.chat_completions_create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                model=self.model,
                max_tokens=1500,
                temperature=0.6
            )
            page_context = resp.choices[0].message.content
        except Exception as e:
            print(f"❌ Error: {e}")
            page_context = f"Error analyzing page: {e}"

        if processed != page_image_path:
            os.remove(processed)

        print(f"\n🔍 Detecting panels...")
        panels, narration_boxes = self.detector.detect_panels(
            page_image_path, extract_text=True, detect_narration=True
        )
        print(f"✓ Found {len(panels)} panels")

        panel_output_dir = os.path.join(output_dir, f"page_{page_number:03d}_panels")

        if not panels:
            print(f"⚠️  No panels detected on page {page_number} — using full page as single panel")
            os.makedirs(panel_output_dir, exist_ok=True)
            from PIL import Image as PILImage
            from greasy.processing.detector import Panel
            img = PILImage.open(page_image_path)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            fallback_path = os.path.join(panel_output_dir, "panel_01.jpg")
            img.save(fallback_path, "JPEG", quality=95)
            w, h = img.size
            fallback_panel = Panel(
                panel_id=1, x=0, y=0, width=w, height=h, area=w * h,
                center_x=w // 2, center_y=h // 2, reading_order=1
            )
            panels = [fallback_panel]
            panel_paths = {1: fallback_path}
        else:
            panel_paths = self.detector.extract_panel_images(page_image_path, panels, panel_output_dir)

        print(f"\n🤖 Analyzing panels with character awareness...")
        panel_analyses = []
        page_characters = set()

        for panel in panels:
            panel_path = panel_paths[panel.panel_id]

            ai_result = self.analyze_panel_with_character_tracking(
                panel_path, panel.panel_id, panel.reading_order,
                page_number, context=page_context
            )

            for char_id in ai_result.get('characters_present', []):
                page_characters.add(char_id)

            panel_analysis = PanelAnalysis(
                panel_id=panel.panel_id,
                reading_order=panel.reading_order,
                analysis=ai_result['analysis'],
                ocr_text=panel.text,
                has_narration=panel.is_narration,
                position={'x': panel.x, 'y': panel.y},
                dimensions={'width': panel.width, 'height': panel.height},
                image_path=panel_path,
                success=ai_result['success'],
                characters_present=ai_result.get('characters_present', []),
                character_descriptions={},
                error=ai_result.get('error')
            )
            panel_analyses.append(panel_analysis)

        page_analysis = PageAnalysis(
            page_number=page_number,
            page_context=page_context,
            total_panels=len(panels),
            panels=panel_analyses,
            narration_boxes=[nb.to_dict() for nb in narration_boxes],
            timestamp=datetime.now().isoformat(),
            success=True,
            characters_on_page=list(page_characters)
        )

        print(f"\n✅ Page {page_number} complete!")
        print(f"   Characters tracked: {len(page_characters)}")
        self.key_manager.print_status()

        return page_analysis

    def save_analysis_with_characters(self, page_analysis: PageAnalysis, output_dir: str = "output"):
        import json
        os.makedirs(output_dir, exist_ok=True)
        page_num = page_analysis.page_number

        json_path = os.path.join(output_dir, f"page_{page_num:03d}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(page_analysis.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"💾 Saved JSON: {json_path}")

        char_path = os.path.join(output_dir, f"page_{page_num:03d}_characters.json")
        page_char_data = {'page_number': page_num, 'characters': []}

        for char_id in page_analysis.characters_on_page:
            char = self.character_tracker.get_character_by_id(char_id)
            if char:
                page_char_data['characters'].append({
                    'character_id': char_id,
                    'name': char.name,
                    'gender': char.gender,
                    'pronouns': char.get_pronouns(),
                    'role': char.role.value,
                    'appearances_on_this_page': len([
                        a for a in char.appearances if a.page_number == page_num
                    ])
                })

        with open(char_path, 'w', encoding='utf-8') as f:
            json.dump(page_char_data, f, indent=2, ensure_ascii=False)
        print(f"👥 Saved character tracking: {char_path}")

        return json_path, char_path

    def save_complete_character_tracking(self, output_dir: str):
        char_file = os.path.join(output_dir, "complete_character_tracking.json")
        self.character_tracker.save_to_file(char_file)
        print(f"\n👥 Saved complete character tracking: {char_file}")