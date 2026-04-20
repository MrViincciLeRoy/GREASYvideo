import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from greasy.core.keys import APIKeyManager, ManagedGroqClient


HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class HFStoryClient:
    """Calls HuggingFace Inference API directly — no model download, pure text."""

    def __init__(self, token: str):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=token)
        print(f"✓ HF story client ready ({HF_MODEL})")

    def generate(self, prompt: str, max_tokens: int = 3000) -> str:
        response = self.client.chat_completion(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


class GroqStoryClient:
    """Groq fallback when no HF token available."""

    def __init__(self, key_manager: APIKeyManager):
        self.client = ManagedGroqClient(key_manager)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        print("⚠ No HF token — story generation using Groq (costs Groq quota)")

    def generate(self, prompt: str, max_tokens: int = 3000) -> str:
        resp = self.client.chat_completions_create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()


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
        text += f"Gender: {self.hero_gender} — pronouns: {pronouns}\n"
        if self.setting:
            text += f"Setting: {self.setting}\n"
        if self.tone:
            text += f"Tone: {self.tone}\n"
        if self.story_theme:
            text += f"Theme: {self.story_theme}\n"
        if self.additional_characters:
            text += "Other Characters:\n"
            for char in self.additional_characters:
                text += f"  - {char['name']}: {char['role']} ({self._get_pronouns(char.get('gender', 'unspecified'))})\n"
        return text

    def _get_pronouns(self, gender: str) -> str:
        return {
            "male": "he/him/his",
            "female": "she/her/hers",
            "non-binary": "they/them/their",
            "unspecified": "they/them/their"
        }.get(gender.lower(), "they/them/their")


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


class ComicStoryGenerator:

    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager

        hf_token = (
            key_manager.huggingface_token
            or os.getenv("HF_TOKEN")
        )

        if hf_token:
            self.story_client = HFStoryClient(hf_token)
            self.using_hf = True
        else:
            self.story_client = GroqStoryClient(key_manager)
            self.using_hf = False

    def generate(self, analysis_path: str, context: StoryContext, output_dir: str) -> Dict:
        print(f"\n{'='*70}")
        print(f"STORY GENERATION — {'HuggingFace API (free, saves Groq quota)' if self.using_hf else 'Groq (fallback)'}")
        print(f"{'='*70}\n")

        with open(analysis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        all_segments = []
        seg_id = 1

        for page_data in data['pages']:
            page_num = page_data['page_number']
            print(f"  Page {page_num}...", end=" ", flush=True)

            segments = self._generate_page(page_data, seg_id, data['title'], context)

            used = {ref.panel_id for seg in segments for ref in seg.panel_references}
            missing = {p['panel_id'] for p in page_data['panels']} - used
            if missing:
                segments.append(self._create_missing_segment(
                    page_data, sorted(missing), seg_id + len(segments), context
                ))

            all_segments.extend(segments)
            seg_id += len(segments)

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
            'generated_via': 'huggingface_api' if self.using_hf else 'groq',
            'segments': [self._seg_to_dict(s) for s in all_segments],
            'generated_at': datetime.now().isoformat()
        }

        self._save_outputs(output, output_dir)

        print(f"\n✓ {len(all_segments)} segments generated via {'HuggingFace' if self.using_hf else 'Groq'}")
        return output

    def _generate_page(self, page_data, start_id, title, context):
        page_num = page_data['page_number']
        panels = page_data['panels']
        page_context = page_data.get('page_context', '')

        panels_text = ""
        for p in panels:
            panels_text += f"\nPanel {p['panel_id']} (Order: {p['reading_order']}):\n"
            panels_text += f"  Visual: {p['analysis'][:500]}\n"
            if p.get('ocr_text'):
                panels_text += f"  Text: {p['ocr_text'][:150]}\n"

        prompt = f"""Write narrative prose adapting "{title}" into a story.

CHARACTER INFO:
{context.to_prompt_text()}

PAGE {page_num} CONTEXT:
{page_context}

PANELS TO ADAPT:
{panels_text}

TASK:
Create 2-4 story segments (100-250 words each) covering all {len(panels)} panels.
- Main character: {context.main_hero}
- Use {context._get_pronouns(context.hero_gender)} pronouns
- Flowing narrative prose only, no markdown or analysis
- Cover panels 1-{len(panels)}

Return ONLY valid JSON array (no backticks, no extra text):
[
  {{
    "segment_text": "Narrative prose here...",
    "panel_ids": [1, 2],
    "narrative_elements": ["scene_description"]
  }}
]"""

        try:
            text = self.story_client.generate(prompt, max_tokens=3000)

            start = text.find('[')
            end = text.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No JSON array in response")

            segments_data = json.loads(text[start:end + 1])
            segments = []

            for i, seg in enumerate(segments_data):
                refs, details = [], []
                for pid in seg['panel_ids']:
                    panel = next((p for p in panels if p['panel_id'] == pid), None)
                    if panel:
                        refs.append(PanelReference(page_num, pid, panel['reading_order']))
                        details.append({'page_number': page_num, 'panel_id': pid,
                                        'reading_order': panel['reading_order']})

                segments.append(StorySegment(
                    segment_id=start_id + i,
                    story_text=seg['segment_text'],
                    panel_references=refs,
                    page_number=page_num,
                    narrative_elements=seg.get('narrative_elements', []),
                    panel_details=details
                ))

            print(f"✓ {len(segments)} segments")
            return segments

        except Exception as e:
            print(f"❌ {e}")
            return self._create_fallback_segment(panels, page_num, start_id, context)

    def _create_fallback_segment(self, panels, page_num, start_id, context):
        refs = [PanelReference(page_num, p['panel_id'], p['reading_order']) for p in panels]
        return [StorySegment(start_id,
                             f"{context.main_hero} continued through the scene.",
                             refs, page_num, ['fallback'], [])]

    def _create_missing_segment(self, page_data, missing_ids, seg_id, context):
        page_num = page_data['page_number']
        panels = [p for p in page_data['panels'] if p['panel_id'] in missing_ids]
        refs = [PanelReference(page_num, p['panel_id'], p['reading_order']) for p in panels]
        return StorySegment(seg_id, f"{context.main_hero} progressed through the scene.",
                            refs, page_num, ['missing'], [])

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
        path = os.path.join(output_dir, "story_complete.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {path}")

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
        print(f"  Saved: {path}")