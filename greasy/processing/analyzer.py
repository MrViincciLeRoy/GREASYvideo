import os
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from greasy.core.keys import APIKeyManager, ManagedGroqClient
from greasy.core.tracker import CharacterTracker, CharacterGuide
from greasy.processing.detector import PanelDetector, Panel


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
                 comic_format: str = "auto", max_workers: int = None,
                 panel_max_size: int = 768, page_context_max_size: int = 1024,
                 skip_page_context: bool = False):
        """
        max_workers:           Parallel panel threads. Defaults to number of Groq keys.
        panel_max_size:        Max image dimension per panel sent to API (was 1024, now 768).
        page_context_max_size: Max image dimension for the full-page context call (was 1536, now 1024).
        skip_page_context:     Skip the one extra API call per page to save requests.
        """
        self.key_manager = key_manager
        self.client = ManagedGroqClient(key_manager)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.character_tracker = CharacterTracker(character_guide)
        self.detector = PanelDetector(comic_format=comic_format)

        self.panel_max_size = panel_max_size
        self.page_context_max_size = page_context_max_size
        self.skip_page_context = skip_page_context
        self.max_workers = max_workers or max(1, len(key_manager.groq_keys))

        # Protects character tracker mutations across threads
        self._tracker_lock = threading.Lock()

        print(f"✓ Comic Analyzer ready")
        print(f"  Protagonist  : {character_guide.protagonist_name or 'Unknown'}")
        print(f"  Format       : {comic_format}")
        print(f"  Workers      : {self.max_workers} parallel panel threads")
        print(f"  Panel size   : ≤{panel_max_size}px  (was 1024px)")
        print(f"  Page context : {'disabled (saving 1 req/page)' if skip_page_context else f'≤{page_context_max_size}px (was 1536px)'}")
        print(f"  Groq keys    : {len(key_manager.groq_keys)}"
              f" + {'HF fallback' if key_manager.huggingface_token else 'no HF'}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _resize_image_if_needed(self, image_path: str, max_size: int) -> str:
        img = Image.open(image_path)
        if max(img.size) <= max_size:
            return image_path

        ratio = max_size / max(img.size)
        new_size = tuple(int(d * ratio) for d in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        base = os.path.splitext(image_path)[0]
        temp_path = base + '_resized.jpg'
        img.save(temp_path, "JPEG", quality=85)
        return temp_path

    def _analyze_panel_worker(self, panel_path: str, panel_id: int,
                               reading_order: int, page_number: int,
                               context: str) -> Dict:
        """
        Runs in a thread pool. Each thread gets its own Groq client via
        ManagedGroqClient's thread-local storage. Only locks for the brief
        character tracker mutation at the end.
        """
        print(f"  🖼️  Panel {panel_id} [{threading.current_thread().name}]")

        try:
            processed = self._resize_image_if_needed(panel_path, self.panel_max_size)
            b64 = self._encode_image(processed)

            # Reading tracker state is safe without a lock (no mutation here)
            char_context = self.character_tracker.generate_prompt_context()

            prompt = f"""You are analyzing Panel #{reading_order} from a comic.

{char_context}

PAGE CONTEXT:
{context}

CRITICAL CHARACTER RULES:
1. Visual cues for protagonist: {', '.join(self.character_tracker.guide.protagonist_visual_cues)}
   → Call them {self.character_tracker.guide.protagonist_name or 'THE PROTAGONIST'}, not "boy/girl/child"
2. Use {self.character_tracker.guide.protagonist_gender} pronouns for the protagonist
3. Check all characters against the guide above

YOUR ANALYSIS MUST INCLUDE:

1. CHARACTERS IDENTIFIED:
   - PROTAGONIST ({self.character_tracker.guide.protagonist_name or 'Name Unknown'}): [present? appearance/expression/action]
   - OTHER CHARACTERS: [names from guide or "Unknown Character X"]

2. DIALOGUE & TEXT:
   - All speech bubbles, narration boxes, sound effects with attribution

3. SCENE & ACTION:
   - What is happening, setting, environment

4. VISUAL ELEMENTS:
   - Art style, framing, colors, mood

5. STORY SIGNIFICANCE:
   - How this advances the narrative"""

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
                temperature=0.4
            )

            analysis_text = resp.choices[0].message.content

            # Lock only for the mutation step
            with self._tracker_lock:
                characters_found = self._extract_characters_from_analysis(
                    analysis_text, page_number, panel_id, reading_order
                )

            if processed != panel_path:
                os.remove(processed)

            return {
                'panel_id': panel_id,
                'reading_order': reading_order,
                'analysis': analysis_text,
                'characters_present': characters_found,
                'success': True,
                'error': None
            }

        except Exception as e:
            print(f"  ❌ Panel {panel_id} error: {e}")
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
        """Must be called while holding self._tracker_lock."""
        characters_found = []
        analysis_lower = analysis.lower()

        protagonist = self.character_tracker.get_protagonist()
        if protagonist:
            visual_match = any(
                cue.lower() in analysis_lower
                for cue in self.character_tracker.guide.protagonist_visual_cues
            )
            if not visual_match:
                terms = (
                    ["young boy", "the boy", "young man", "he appears", "he is",
                     "he has", "his hair", "his eyes", "his expression",
                     "the protagonist", "main character"]
                    if protagonist.gender == "male" else
                    ["young girl", "the girl", "young woman", "she appears", "she is",
                     "she has", "her hair", "her eyes", "her expression",
                     "the protagonist", "main character"]
                )
                visual_match = any(t in analysis_lower for t in terms)

            if visual_match:
                self.character_tracker.track_character_appearance(
                    character_id=protagonist.character_id,
                    page=page, panel=panel, reading_order=reading_order,
                    description=analysis[:200]
                )
                characters_found.append(protagonist.character_id)
                print(f"      ✓ Protagonist in panel {panel}")

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
                print(f"      ✓ {character.name} in panel {panel}")

        return characters_found

    def analyze_complete_page(self, page_image_path: str, page_number: int,
                               output_dir: str = "output") -> PageAnalysis:
        print(f"\n{'='*70}")
        print(f"📖 Page {page_number}")
        print(f"{'='*70}")

        # ── Optional full-page context call ───────────────────────────────
        if self.skip_page_context:
            page_context = ""
            print("  ⏭️  Page context skipped")
        else:
            print(f"  🔍 Full-page context (≤{self.page_context_max_size}px)...")
            processed = self._resize_image_if_needed(page_image_path, self.page_context_max_size)
            b64 = self._encode_image(processed)

            prompt = f"""Analyze this comic page:

{self.character_tracker.generate_prompt_context()}

- Overall story beat and narrative
- Characters present (use correct names from guide)
- Setting and emotional tone
- Key events

Be concise."""

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
                    max_tokens=800,
                    temperature=0.3
                )
                page_context = resp.choices[0].message.content
            except Exception as e:
                print(f"  ❌ Page context error: {e}")
                page_context = ""

            if processed != page_image_path:
                os.remove(processed)

        # ── Panel detection ────────────────────────────────────────────────
        print(f"  🔍 Detecting panels...")
        panels, narration_boxes = self.detector.detect_panels(
            page_image_path, extract_text=True, detect_narration=True
        )
        print(f"  ✓ {len(panels)} panels")

        panel_output_dir = os.path.join(output_dir, f"page_{page_number:03d}_panels")

        if not panels:
            print(f"  ⚠️  No panels detected — treating full page as panel 1")
            os.makedirs(panel_output_dir, exist_ok=True)
            img = Image.open(page_image_path)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            fallback_path = os.path.join(panel_output_dir, "panel_01.jpg")
            img.save(fallback_path, "JPEG", quality=95)
            w, h = img.size
            panels = [Panel(
                panel_id=1, x=0, y=0, width=w, height=h, area=w * h,
                center_x=w // 2, center_y=h // 2, reading_order=1
            )]
            panel_paths = {1: fallback_path}
        else:
            panel_paths = self.detector.extract_panel_images(
                page_image_path, panels, panel_output_dir
            )

        # ── Parallel panel analysis ────────────────────────────────────────
        workers = min(self.max_workers, len(panels))
        print(f"\n  🤖 Analyzing {len(panels)} panels with {workers} threads...")

        futures = {}
        results_by_id = {}

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="panel") as pool:
            for panel in panels:
                future = pool.submit(
                    self._analyze_panel_worker,
                    panel_paths[panel.panel_id],
                    panel.panel_id,
                    panel.reading_order,
                    page_number,
                    page_context
                )
                futures[future] = panel

            for future in as_completed(futures):
                panel = futures[future]
                try:
                    results_by_id[panel.panel_id] = future.result()
                except Exception as e:
                    results_by_id[panel.panel_id] = {
                        'panel_id': panel.panel_id,
                        'reading_order': panel.reading_order,
                        'analysis': "",
                        'characters_present': [],
                        'success': False,
                        'error': str(e)
                    }

        # ── Reassemble in reading order ────────────────────────────────────
        panel_analyses = []
        page_characters = set()

        for panel in sorted(panels, key=lambda p: p.reading_order):
            ai_result = results_by_id[panel.panel_id]
            for char_id in ai_result.get('characters_present', []):
                page_characters.add(char_id)
            panel_analyses.append(PanelAnalysis(
                panel_id=panel.panel_id,
                reading_order=panel.reading_order,
                analysis=ai_result['analysis'],
                ocr_text=panel.text,
                has_narration=panel.is_narration,
                position={'x': panel.x, 'y': panel.y},
                dimensions={'width': panel.width, 'height': panel.height},
                image_path=panel_paths[panel.panel_id],
                success=ai_result['success'],
                characters_present=ai_result.get('characters_present', []),
                character_descriptions={},
                error=ai_result.get('error')
            ))

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

        print(f"\n  ✅ Page {page_number} — {len(panel_analyses)} panels, "
              f"{len(page_characters)} characters tracked")
        self.key_manager.print_status()

        return page_analysis

    def save_analysis_with_characters(self, page_analysis: PageAnalysis, output_dir: str = "output"):
        import json
        os.makedirs(output_dir, exist_ok=True)
        page_num = page_analysis.page_number

        json_path = os.path.join(output_dir, f"page_{page_num:03d}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(page_analysis.to_dict(), f, indent=2, ensure_ascii=False)

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

        print(f"  💾 {json_path}")
        print(f"  👥 {char_path}")
        return json_path, char_path

    def save_complete_character_tracking(self, output_dir: str):
        char_file = os.path.join(output_dir, "complete_character_tracking.json")
        self.character_tracker.save_to_file(char_file)
        print(f"\n👥 Saved: {char_file}")
