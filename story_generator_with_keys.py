
# story_generator_with_keys.py
"""
Story Generator with API Key Rotation
"""

import os
import json
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

# Import key manager
from enhanced_api_key_manager import APIKeyManager, ManagedGroqClient

# ============================================================================
# CONFIGURATION
# ============================================================================

ANALYSIS_PATH = "/content/knight_king_analysis/complete_analysis.json"
OUTPUT_DIR = "story_output"

# API KEYS
GROQ_KEYS = [
            os.getenv('GROQ_KEY'), 
        os.getenv('GROQ_KEY_2'), 
       os.getenv('GROQ_KEY_3'), 
       os.getenv('GROQ_KEY_4')
            # Add more keys for higher capacity
]

# Character Configuration
MAIN_HERO = "Leon"
HERO_ROLE = "Knight King"
HERO_GENDER = "male"
SETTING = "Fantasy kingdom"
TONE = "epic fantasy"
STORY_THEME = "rebirth and redemption"

ADDITIONAL_CHARACTERS = [
    {"name": "Nanny", "role": "Caretaker", "gender": "female"},
    {"name": "Grand Duke", "role": "Noble Lord", "gender": "male"}
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StoryContext:
    main_hero: str = "the protagonist"
    hero_role: str = "hero"
    hero_gender: str = "male"
    setting: str = ""
    tone: str = "adventure"
    additional_characters: List[Dict[str, str]] = None
    story_theme: str = ""

    def to_prompt_text(self) -> str:
        pronouns = self._get_pronouns(self.hero_gender)
        text = f"Main Hero: {self.main_hero} ({self.hero_role})\n"
        text += f"Gender: {self.hero_gender} - Use pronouns: {pronouns}\n"
        if self.setting:
            text += f"Setting: {self.setting}\n"
        if self.tone:
            text += f"Tone: {self.tone}\n"
        if self.story_theme:
            text += f"Theme: {self.story_theme}\n"
        if self.additional_characters:
            text += "Other Characters:\n"
            for char in self.additional_characters:
                char_pronouns = self._get_pronouns(char.get('gender', 'unspecified'))
                text += f"  - {char['name']}: {char['role']} ({char_pronouns})\n"
        return text

    def _get_pronouns(self, gender: str) -> str:
        pronoun_map = {
            "male": "he/him/his",
            "female": "she/her/hers",
            "non-binary": "they/them/their",
            "unspecified": "they/them/their"
        }
        return pronoun_map.get(gender.lower(), "they/them/their")

@dataclass
class PanelReference:
    page_number: int
    panel_id: int
    reading_order: int

@dataclass
class StorySegment:
    segment_id: int
    story_text: str
    panel_references: List[PanelReference]
    page_number: int
    narrative_elements: List[str]
    panel_details: List[Dict] = None

# ============================================================================
# STORY GENERATOR
# ============================================================================

class ComicStoryGenerator:
    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
        self.client = ManagedGroqClient(key_manager)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"

    def generate(self, analysis_path: str, context: StoryContext, output_dir: str) -> Dict:
        print(f"\n{'='*80}")
        print("📖 GENERATING STORY WITH API KEY ROTATION")
        print(f"{'='*80}\n")

        # Show initial API status
        self.key_manager.print_status()

        with open(analysis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        all_segments = []
        seg_id = 1

        for page_data in data['pages']:
            page_num = page_data['page_number']
            print(f"\n📄 Page {page_num}...")

            # Show API status every 5 pages
            if page_num % 5 == 0:
                self.key_manager.print_status()

            segments = self._generate_page(page_data, seg_id, data['title'], context)

            # Check coverage
            used = {ref.panel_id for seg in segments for ref in seg.panel_references}
            all_panels = {p['panel_id'] for p in page_data['panels']}
            missing = all_panels - used

            if missing:
                print(f"   ⚠ Missing panels {sorted(missing)}, creating segment...")
                extra = self._create_missing_segment(page_data, sorted(missing), seg_id + len(segments), context)
                segments.append(extra)

            all_segments.extend(segments)
            seg_id += len(segments)

        # Create output
        output = {
            'title': data['title'],
            'story_context': {
                'main_hero': context.main_hero,
                'hero_role': context.hero_role,
                'hero_gender': context.hero_gender,
                'setting': context.setting,
                'tone': context.tone,
                'story_theme': context.story_theme,
                'additional_characters': context.additional_characters or []
            },
            'total_pages': data['total_pages'],
            'total_segments': len(all_segments),
            'segments': [self._seg_to_dict(s) for s in all_segments],
            'generated_at': datetime.now().isoformat()
        }

        self._save_outputs(output, output_dir)

        # Show final API usage
        print(f"\n{'='*80}")
        print("📊 Final API Usage:")
        print(f"{'='*80}")
        self.key_manager.print_status()

        return output

    def _generate_page(self, page_data, start_id, title, context):
        page_num = page_data['page_number']
        panels = page_data['panels']
        page_context = page_data.get('page_context', '')

        # Extract panel descriptions
        panels_descriptions = []
        for p in panels:
            desc = f"\nPanel {p['panel_id']} (Order: {p['reading_order']}):\n"
            desc += f"  Visual: {p['analysis'][:500]}\n"

            if p.get('ocr_text'):
                desc += f"  Text: {p['ocr_text'][:150]}\n"

            panels_descriptions.append(desc)

        panels_text = "".join(panels_descriptions)

        prompt = f"""Write narrative prose adapting "{title}" into a story.

CHARACTER INFO:
{context.to_prompt_text()}

PAGE {page_num} CONTEXT:
{page_context}

PANELS TO ADAPT:
{panels_text}

TASK:
Create 2-4 story segments (100-250 words each) that narratively describe these {len(panels)} panels.
- Main character: {context.main_hero}
- Use {context._get_pronouns(context.hero_gender)} pronouns
- Write flowing narrative prose (no markdown, no analysis formatting)
- Cover all panels (1-{len(panels)})

Return ONLY JSON (no backticks):
[
  {{
    "segment_text": "Narrative prose here...",
    "panel_ids": [1, 2],
    "narrative_elements": ["scene_description"]
  }}
]"""

        try:
            resp = self.client.chat_completions_create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=3000,
                temperature=0.7
            )

            text = resp.choices[0].message.content.strip()

            # Extract JSON
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                text = text[start:end+1]

            segments_data = json.loads(text)

            # Build segments
            segments = []
            for i, seg in enumerate(segments_data):
                refs = []
                details = []

                for pid in seg['panel_ids']:
                    panel = next((p for p in panels if p['panel_id'] == pid), None)
                    if panel:
                        refs.append(PanelReference(page_num, pid, panel['reading_order']))
                        details.append({
                            'page_number': page_num,
                            'panel_id': pid,
                            'reading_order': panel['reading_order']
                        })

                segments.append(StorySegment(
                    segment_id=start_id + i,
                    story_text=seg['segment_text'],
                    panel_references=refs,
                    page_number=page_num,
                    narrative_elements=seg.get('narrative_elements', []),
                    panel_details=details
                ))

            print(f"   ✓ Generated {len(segments)} segments")
            return segments

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return self._create_fallback_segment(panels, page_num, start_id, context)

    def _create_fallback_segment(self, panels, page_num, start_id, context):
        """Create fallback segment"""
        refs = [PanelReference(page_num, p['panel_id'], p['reading_order']) for p in panels]
        text = f"{context.main_hero} continued through the events depicted in the panels."

        return [StorySegment(start_id, text, refs, page_num, ['fallback'], [])]

    def _create_missing_segment(self, page_data, missing_ids, seg_id, context):
        """Create segment for missing panels"""
        page_num = page_data['page_number']
        panels = [p for p in page_data['panels'] if p['panel_id'] in missing_ids]

        refs = [PanelReference(page_num, p['panel_id'], p['reading_order']) for p in panels]
        text = f"{context.main_hero} progressed through the scene."

        return StorySegment(seg_id, text, refs, page_num, ['missing'], [])

    def _seg_to_dict(self, seg):
        return {
            'segment_id': seg.segment_id,
            'story_text': seg.story_text,
            'panel_references': [
                {'page': r.page_number, 'panel_id': r.panel_id, 'reading_order': r.reading_order}
                for r in seg.panel_references
            ],
            'page_number': seg.page_number,
            'narrative_elements': seg.narrative_elements
        }

    def _save_outputs(self, output, output_dir):
        # Main JSON
        path = os.path.join(output_dir, "story_complete.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved: {path}")

        # Panel mapping for video
        mapping = {
            'title': output['title'],
            'story_context': output['story_context'],
            'total_segments': output['total_segments'],
            'total_pages': output['total_pages'],
            'story_segments': [
                {
                    'segment_id': s['segment_id'],
                    'page_number': s['page_number'],
                    'story_paragraph': s['story_text'],
                    'panels': s['panel_references'],
                    'panel_count': len(s['panel_references'])
                } for s in output['segments']
            ]
        }

        path = os.path.join(output_dir, "panel_to_story_mapping.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
        print(f"🗺️  Saved: {path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║        STORY GENERATOR WITH API KEY ROTATION                 ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Validate
    if not os.path.exists(ANALYSIS_PATH):
        print(f"❌ Error: {ANALYSIS_PATH} not found!")
        exit(1)

    # Create key manager
    key_manager = APIKeyManager(
        groq_keys=GROQ_KEYS,
        huggingface_token=None,
        state_file="story_api_keys.json"
    )

    # Create context
    context = StoryContext(
        main_hero=MAIN_HERO,
        hero_role=HERO_ROLE,
        hero_gender=HERO_GENDER,
        setting=SETTING,
        tone=TONE,
        additional_characters=ADDITIONAL_CHARACTERS,
        story_theme=STORY_THEME
    )

    # Generate
    generator = ComicStoryGenerator(key_manager)
    result = generator.generate(ANALYSIS_PATH, context, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 {result['total_segments']} segments created")
    print(f"📄 {result['total_pages']} pages processed")
    print(f"\n🎬 Ready for video generation!")