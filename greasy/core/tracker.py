
# character_tracker.py

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum

class CharacterRole(Enum):
    """Character importance levels"""
    PROTAGONIST = "protagonist"
    MAIN_CHARACTER = "main_character"
    SUPPORTING = "supporting"
    MINOR = "minor"
    UNKNOWN = "unknown"

@dataclass
class CharacterAppearance:
    """Single appearance of a character in a panel"""
    page_number: int
    panel_id: int
    reading_order: int
    description: str
    confidence: float = 0.0
    dialogue: Optional[str] = None

@dataclass
class Character:
    """Complete character profile with tracking"""
    # Identity
    character_id: str  # Unique ID (e.g., "char_001")
    name: Optional[str] = None  # Might not know initially
    aliases: List[str] = field(default_factory=list)  # "the boy", "young master", etc.

    # Demographics
    gender: str = "unspecified"  # male, female, non-binary, unspecified
    age_category: str = "unknown"  # child, teen, adult, elderly, unknown

    # Role & Characteristics
    role: CharacterRole = CharacterRole.UNKNOWN
    description: str = ""  # Physical appearance, clothing, notable features
    personality_traits: List[str] = field(default_factory=list)

    # Tracking
    first_appearance: Optional[Dict] = None  # {page, panel, reading_order}
    appearances: List[CharacterAppearance] = field(default_factory=list)
    total_appearances: int = 0
    pages_seen_on: Set[int] = field(default_factory=set)

    # Relationships
    relationships: Dict[str, str] = field(default_factory=dict)  # {char_id: relationship}

    # Metadata
    confidence: float = 0.0  # How confident we are this is the same character
    notes: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_pronouns(self) -> str:
        """Get appropriate pronouns for this character"""
        pronoun_map = {
            "male": "he/him/his",
            "female": "she/her/hers",
            "non-binary": "they/them/their",
            "unspecified": "they/them/their"
        }
        return pronoun_map.get(self.gender.lower(), "they/them/their")

    def add_appearance(self, appearance: CharacterAppearance):
        """Record a new appearance"""
        self.appearances.append(appearance)
        self.total_appearances += 1
        self.pages_seen_on.add(appearance.page_number)

        if self.first_appearance is None:
            self.first_appearance = {
                'page': appearance.page_number,
                'panel': appearance.panel_id,
                'reading_order': appearance.reading_order
            }

        self.last_updated = datetime.now().isoformat()

    def add_alias(self, alias: str):
        """Add a new way this character is referred to"""
        if alias and alias not in self.aliases:
            self.aliases.append(alias)
            self.last_updated = datetime.now().isoformat()

    def set_name(self, name: str):
        """Set or update character name"""
        if name:
            self.name = name
            self.add_alias(name)
            self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'character_id': self.character_id,
            'name': self.name,
            'aliases': self.aliases,
            'gender': self.gender,
            'age_category': self.age_category,
            'role': self.role.value,
            'description': self.description,
            'personality_traits': self.personality_traits,
            'pronouns': self.get_pronouns(),
            'first_appearance': self.first_appearance,
            'total_appearances': self.total_appearances,
            'pages_seen_on': sorted(list(self.pages_seen_on)),
            'appearances': [
                {
                    'page': a.page_number,
                    'panel': a.panel_id,
                    'reading_order': a.reading_order,
                    'description': a.description,
                    'dialogue': a.dialogue,
                    'confidence': a.confidence
                } for a in self.appearances
            ],
            'relationships': self.relationships,
            'confidence': self.confidence,
            'notes': self.notes,
            'last_updated': self.last_updated
        }

@dataclass
class CharacterGuide:
    """
    Initial character guide provided by user.
    This sets the ground truth for known characters.
    """
    # Main protagonist
    protagonist_name: Optional[str] = None
    protagonist_gender: str = "unspecified"
    protagonist_role: str = "protagonist"
    protagonist_description: str = ""

    # Supporting cast (known upfront)
    known_characters: List[Dict[str, str]] = field(default_factory=list)
    # Format: [{'name': 'X', 'gender': 'Y', 'role': 'Z', 'description': '...'}]

    # Context
    setting: str = ""
    tone: str = ""
    story_theme: str = ""

    # Recognition hints
    protagonist_visual_cues: List[str] = field(default_factory=list)
    # e.g., ["blonde hair", "green tunic", "always carries sword"]

    def to_dict(self) -> Dict:
        return {
            'protagonist': {
                'name': self.protagonist_name,
                'gender': self.protagonist_gender,
                'role': self.protagonist_role,
                'description': self.protagonist_description,
                'visual_cues': self.protagonist_visual_cues
            },
            'known_characters': self.known_characters,
            'context': {
                'setting': self.setting,
                'tone': self.tone,
                'story_theme': self.story_theme
            }
        }

class CharacterTracker:
    """
    Tracks characters throughout the comic analysis.
    Maintains consistency and resolves identities across pages.
    """

    def __init__(self, guide: CharacterGuide):
        self.guide = guide
        self.characters: Dict[str, Character] = {}
        self.next_char_id = 1

        # Initialize protagonist if provided
        if guide.protagonist_name:
            self._initialize_protagonist()

        # Initialize known characters
        for char_info in guide.known_characters:
            self._initialize_known_character(char_info)

    def _initialize_protagonist(self):
        """Create protagonist character entry"""
        char_id = "char_protagonist"

        protagonist = Character(
            character_id=char_id,
            name=self.guide.protagonist_name,
            aliases=[self.guide.protagonist_name] if self.guide.protagonist_name else [],
            gender=self.guide.protagonist_gender,
            role=CharacterRole.PROTAGONIST,
            description=self.guide.protagonist_description,
            confidence=1.0  # We're certain about the protagonist
        )

        # Add common aliases that might appear before we know the name
        if self.guide.protagonist_gender == "male":
            protagonist.add_alias("the boy")
            protagonist.add_alias("young man")
            protagonist.add_alias("he")
        elif self.guide.protagonist_gender == "female":
            protagonist.add_alias("the girl")
            protagonist.add_alias("young woman")
            protagonist.add_alias("she")

        self.characters[char_id] = protagonist

    def _initialize_known_character(self, char_info: Dict):
        """Initialize a character we know about from the guide"""
        char_id = f"char_{self.next_char_id:03d}"
        self.next_char_id += 1

        # Determine role
        role_str = char_info.get('role', 'supporting').lower()
        if 'main' in role_str or 'protagonist' in role_str:
            role = CharacterRole.MAIN_CHARACTER
        elif 'support' in role_str:
            role = CharacterRole.SUPPORTING
        else:
            role = CharacterRole.UNKNOWN

        character = Character(
            character_id=char_id,
            name=char_info.get('name'),
            aliases=[char_info.get('name')] if char_info.get('name') else [],
            gender=char_info.get('gender', 'unspecified'),
            role=role,
            description=char_info.get('description', ''),
            confidence=0.9  # High confidence since it's from the guide
        )

        self.characters[char_id] = character

    def get_protagonist(self) -> Optional[Character]:
        """Get the protagonist character"""
        return self.characters.get("char_protagonist")

    def identify_character(self, description: str, page: int, panel: int,
                          dialogue: Optional[str] = None) -> Optional[str]:
        """
        Try to identify which character this is based on description.
        Returns character_id if match found, None otherwise.

        This is where AI vision analysis text gets matched to known characters.
        """
        description_lower = description.lower()

        # First, check protagonist with visual cues
        protagonist = self.get_protagonist()
        if protagonist:
            # Check if description matches protagonist visual cues
            if any(cue.lower() in description_lower
                   for cue in self.guide.protagonist_visual_cues):
                return protagonist.character_id

            # Check gender-based descriptions
            if protagonist.gender == "male":
                male_terms = ["boy", "young man", "he ", "his ", "him "]
                if any(term in description_lower for term in male_terms):
                    return protagonist.character_id
            elif protagonist.gender == "female":
                female_terms = ["girl", "young woman", "she ", "her ", "hers "]
                if any(term in description_lower for term in female_terms):
                    return protagonist.character_id

        # Check known characters by name or description
        for char_id, character in self.characters.items():
            if character.name and character.name.lower() in description_lower:
                return char_id

            # Check aliases
            if any(alias.lower() in description_lower for alias in character.aliases):
                return char_id

        return None

    def track_character_appearance(self, character_id: str, page: int,
                                   panel: int, reading_order: int,
                                   description: str, dialogue: Optional[str] = None):
        """Record a character appearance"""
        if character_id not in self.characters:
            return

        appearance = CharacterAppearance(
            page_number=page,
            panel_id=panel,
            reading_order=reading_order,
            description=description,
            dialogue=dialogue,
            confidence=0.8
        )

        self.characters[character_id].add_appearance(appearance)

    def create_unknown_character(self, description: str, page: int,
                                 panel: int, reading_order: int,
                                 gender: str = "unspecified") -> str:
        """
        Create a new character entry for an unidentified character.
        Returns the new character_id.
        """
        char_id = f"char_{self.next_char_id:03d}"
        self.next_char_id += 1

        character = Character(
            character_id=char_id,
            gender=gender,
            role=CharacterRole.UNKNOWN,
            description=description,
            confidence=0.5  # Low confidence for unknown
        )

        # Add first appearance
        appearance = CharacterAppearance(
            page_number=page,
            panel_id=panel,
            reading_order=reading_order,
            description=description
        )
        character.add_appearance(appearance)

        self.characters[char_id] = character
        return char_id

    def update_character_name(self, character_id: str, name: str):
        """Update character name when it's revealed"""
        if character_id in self.characters:
            self.characters[character_id].set_name(name)

    def merge_characters(self, source_id: str, target_id: str):
        """
        Merge two character entries (when we realize they're the same person).
        Moves all appearances from source to target.
        """
        if source_id not in self.characters or target_id not in self.characters:
            return

        source = self.characters[source_id]
        target = self.characters[target_id]

        # Transfer appearances
        target.appearances.extend(source.appearances)
        target.total_appearances += source.total_appearances
        target.pages_seen_on.update(source.pages_seen_on)

        # Merge aliases
        for alias in source.aliases:
            target.add_alias(alias)

        # Remove source
        del self.characters[source_id]

    def get_character_by_id(self, character_id: str) -> Optional[Character]:
        """Get character by ID"""
        return self.characters.get(character_id)

    def get_all_characters(self) -> List[Character]:
        """Get all tracked characters"""
        return list(self.characters.values())

    def get_characters_by_role(self, role: CharacterRole) -> List[Character]:
        """Get all characters with a specific role"""
        return [c for c in self.characters.values() if c.role == role]

    def get_character_summary(self) -> Dict:
        """Get summary of all characters"""
        return {
            'total_characters': len(self.characters),
            'protagonist': self.get_protagonist().to_dict() if self.get_protagonist() else None,
            'main_characters': [c.to_dict() for c in self.get_characters_by_role(CharacterRole.MAIN_CHARACTER)],
            'supporting_characters': [c.to_dict() for c in self.get_characters_by_role(CharacterRole.SUPPORTING)],
            'unknown_characters': [c.to_dict() for c in self.get_characters_by_role(CharacterRole.UNKNOWN)]
        }

    def generate_prompt_context(self) -> str:
        """
        Generate context text for AI prompts to maintain character consistency.
        """
        context = "CHARACTER GUIDE - USE THESE NAMES AND PRONOUNS:\n\n"

        # Protagonist
        protag = self.get_protagonist()
        if protag:
            context += f"PROTAGONIST: {protag.name or 'Unknown name'}\n"
            context += f"  Gender: {protag.gender}\n"
            context += f"  Pronouns: {protag.get_pronouns()}\n"
            context += f"  Description: {protag.description}\n"
            if self.guide.protagonist_visual_cues:
                context += f"  Visual Cues: {', '.join(self.guide.protagonist_visual_cues)}\n"
            context += f"  Common references: {', '.join(protag.aliases[:5])}\n\n"

        # Main characters
        main_chars = self.get_characters_by_role(CharacterRole.MAIN_CHARACTER)
        if main_chars:
            context += "MAIN CHARACTERS:\n"
            for char in main_chars:
                context += f"  - {char.name or 'Unknown'} ({char.gender}, {char.get_pronouns()})\n"
                if char.description:
                    context += f"    {char.description}\n"
            context += "\n"

        # Supporting characters
        supporting = self.get_characters_by_role(CharacterRole.SUPPORTING)
        if supporting:
            context += "SUPPORTING CHARACTERS:\n"
            for char in supporting[:5]:  # Limit to first 5
                context += f"  - {char.name or 'Unknown'} ({char.gender})\n"
            context += "\n"

        context += "CRITICAL INSTRUCTIONS:\n"
        context += "- When you see descriptions like 'young boy' or 'the girl', "
        context += f"these refer to {protag.name if protag and protag.name else 'the protagonist'}\n"
        context += f"- Always use {protag.get_pronouns() if protag else 'appropriate'} pronouns\n"
        context += "- If a name is revealed in dialogue, note it for future reference\n"
        context += "- Maintain character consistency across panels\n"

        return context

    def save_to_file(self, filepath: str):
        """Save character tracking data to JSON"""
        data = {
            'guide': self.guide.to_dict(),
            'characters': {cid: char.to_dict() for cid, char in self.characters.items()},
            'summary': self.get_character_summary(),
            'generated_at': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CharacterTracker':
        """Load character tracking data from JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct guide
        guide_data = data['guide']
        guide = CharacterGuide(
            protagonist_name=guide_data['protagonist']['name'],
            protagonist_gender=guide_data['protagonist']['gender'],
            protagonist_role=guide_data['protagonist']['role'],
            protagonist_description=guide_data['protagonist']['description'],
            protagonist_visual_cues=guide_data['protagonist']['visual_cues'],
            known_characters=guide_data['known_characters'],
            setting=guide_data['context']['setting'],
            tone=guide_data['context']['tone'],
            story_theme=guide_data['context']['story_theme']
        )

        tracker = cls(guide)

        # Load all characters
        tracker.characters = {}
        for char_id, char_data in data['characters'].items():
            character = Character(
                character_id=char_data['character_id'],
                name=char_data['name'],
                aliases=char_data['aliases'],
                gender=char_data['gender'],
                age_category=char_data['age_category'],
                role=CharacterRole(char_data['role']),
                description=char_data['description'],
                personality_traits=char_data['personality_traits'],
                first_appearance=char_data['first_appearance'],
                total_appearances=char_data['total_appearances'],
                pages_seen_on=set(char_data['pages_seen_on']),
                relationships=char_data['relationships'],
                confidence=char_data['confidence'],
                notes=char_data['notes'],
                last_updated=char_data['last_updated']
            )

            # Reconstruct appearances
            for app_data in char_data['appearances']:
                appearance = CharacterAppearance(
                    page_number=app_data['page'],
                    panel_id=app_data['panel'],
                    reading_order=app_data['reading_order'],
                    description=app_data['description'],
                    dialogue=app_data.get('dialogue'),
                    confidence=app_data.get('confidence', 0.0)
                )
                character.appearances.append(appearance)

            tracker.characters[char_id] = character

        return tracker


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a character guide
    guide = CharacterGuide(
        protagonist_name="Sir Eldric",
        protagonist_gender="male",
        protagonist_role="Knight Commander",
        protagonist_description="A battle-worn knight in his thirties with silver armor",
        protagonist_visual_cues=[
            "silver armor",
            "blue cape",
            "scar across left cheek",
            "carries a broadsword"
        ],
        known_characters=[
            {
                'name': 'Princess Aria',
                'gender': 'female',
                'role': 'Supporting',
                'description': 'Young princess in royal robes, blonde hair'
            },
            {
                'name': 'Commander Vex',
                'gender': 'male',
                'role': 'Supporting',
                'description': 'Antagonist, dark armor, red cape'
            }
        ],
        setting="Medieval fantasy kingdom at war",
        tone="Epic and dramatic",
        story_theme="Honor and sacrifice"
    )

    # Initialize tracker
    tracker = CharacterTracker(guide)

    # Simulate tracking a character appearance
    # (This would normally come from AI vision analysis)
    description = "A young knight in silver armor with a blue cape stands ready"

    # Try to identify
    char_id = tracker.identify_character(
        description=description,
        page=1,
        panel=1,
        dialogue="For honor!"
    )

    if char_id:
        # Record the appearance
        tracker.track_character_appearance(
            character_id=char_id,
            page=1,
            panel=1,
            reading_order=1,
            description=description,
            dialogue="For honor!"
        )
        print(f"✓ Identified as: {tracker.get_character_by_id(char_id).name}")

    # Generate prompt context for AI
    print("\n" + "="*80)
    print(tracker.generate_prompt_context())
    print("="*80)

    # Save tracking data
    tracker.save_to_file("character_tracking.json")
    print("\n✓ Saved character tracking data")

    # Show summary
    summary = tracker.get_character_summary()
    print(f"\n📊 Total characters tracked: {summary['total_characters']}")
    print(f"   Protagonist: {summary['protagonist']['name']}")
    print(f"   Main characters: {len(summary['main_characters'])}")
    print(f"   Supporting: {len(summary['supporting_characters'])}")