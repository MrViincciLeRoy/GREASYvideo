#!/usr/bin/env bash
# Run from your GREASYvideo project root: bash migrate.sh

set -e

echo "Creating package directories..."
mkdir -p greasy/core greasy/processing greasy/story greasy/video

# ── __init__.py files ─────────────────────────────────────────────────────────
touch greasy/__init__.py greasy/core/__init__.py \
      greasy/processing/__init__.py greasy/story/__init__.py \
      greasy/video/__init__.py

# ── Move files ────────────────────────────────────────────────────────────────
mv enhanced_api_key_manager.py          greasy/core/keys.py
mv character_tracker.py                 greasy/core/tracker.py
mv panel_detector.py                    greasy/processing/detector.py
mv integrated_analyzer_with_key_manager.py greasy/processing/analyzer.py
mv pdf_comic_processor_with_keys.py     greasy/processing/pdf.py
mv story_generator_with_keys.py         greasy/story/generator.py
mv VidGen.py                            greasy/video/generator.py
mv master_comic_pipeline.py             pipeline.py

echo "Fixing imports..."

# greasy/processing/analyzer.py
sed -i \
  -e 's/from enhanced_api_key_manager import/from greasy.core.keys import/' \
  -e 's/from character_tracker import/from greasy.core.tracker import/' \
  -e 's/from panel_detector import/from greasy.processing.detector import/' \
  greasy/processing/analyzer.py

# greasy/processing/pdf.py
sed -i \
  -e 's/from enhanced_api_key_manager import/from greasy.core.keys import/' \
  -e 's/from character_tracker import/from greasy.core.tracker import/' \
  -e 's/from integrated_analyzer_with_key_manager import/from greasy.processing.analyzer import/' \
  greasy/processing/pdf.py

# greasy/story/generator.py
sed -i \
  -e 's/from enhanced_api_key_manager import/from greasy.core.keys import/' \
  greasy/story/generator.py

# pipeline.py
sed -i \
  -e 's/from enhanced_api_key_manager import/from greasy.core.keys import/' \
  -e 's/from character_tracker import/from greasy.core.tracker import/' \
  -e 's/from pdf_comic_processor_with_keys import/from greasy.processing.pdf import/' \
  -e 's/from story_generator_with_keys import/from greasy.story.generator import/' \
  -e 's/from VidGen import/from greasy.video.generator import/' \
  pipeline.py

echo ""
echo "Done. New structure:"
find greasy -name "*.py" | sort
echo "pipeline.py"
echo ""
echo "Run the pipeline the same way:"
echo "  python pipeline.py data/MyComic.pdf --start-page 1 --end-page 10"
