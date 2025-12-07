
"""
Memory-Efficient Video Generator with Kokoro TTS
Uses Kokoro open-weight TTS model for high-quality audio narration
"""

import json
import os
import gc
from pathlib import Path
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, VideoFileClip, concatenate_audioclips
import tempfile
import shutil
import soundfile as sf
import torch
import numpy as np

class KokoroTTSVideoGenerator:
    def __init__(self, json_path, base_panels_folder, output_path="output_video.mp4", 
                 include_audio=True, batch_size=12, voice='af_heart', lang_code='a'):
        """
        Initialize the video generator with Kokoro TTS
        
        Args:
            json_path: Path to panel_to_story_mapping.json
            base_panels_folder: Base folder containing page_XXX_panels subfolders
            output_path: Path for output video file
            include_audio: Whether to include audio narration
            batch_size: Number of segments to process before creating intermediate video
            voice: Kokoro voice to use (af_heart, af_sky, af_bella, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis)
            lang_code: Language code ('a' for American English, 'b' for British English)
        """
        self.json_path = json_path
        self.base_panels_folder = Path(base_panels_folder)
        self.output_path = output_path
        self.temp_dir = tempfile.mkdtemp()
        self.batch_temp_dir = os.path.join(self.temp_dir, "batch_videos")
        os.makedirs(self.batch_temp_dir, exist_ok=True)
        
        self.last_used_panel = None
        self.include_audio = include_audio
        self.batch_size = batch_size
        self.voice = voice
        self.lang_code = lang_code
        self.sample_rate = 24000
        
        # Kokoro has a limit on text length (~500 chars recommended for best quality)
        # We'll split longer texts into chunks
        self.max_chars_per_chunk = 400

        # Available voices for reference
        self.available_voices = {
            'af': ['af_heart', 'af_sky', 'af_bella'],  # American Female
            'am': ['am_adam', 'am_michael'],           # American Male
            'bf': ['bf_emma', 'bf_isabella'],          # British Female
            'bm': ['bm_george', 'bm_lewis']            # British Male
        }

        # Initialize Kokoro pipeline
        if self.include_audio:
            print(f"🎙️  Loading Kokoro TTS (voice: {voice}, lang: {lang_code})...")
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=lang_code)
            print("✓ Kokoro TTS loaded successfully!")
        else:
            self.pipeline = None

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print("JSON keys:", list(self.data.keys()))
        if 'story_segments' in self.data:
            total_segments = len(self.data['story_segments'])
            print(f"Found {total_segments} segments")
            print(f"Batch size: {batch_size}")
            print(f"TTS Voice: {voice}")
            print(f"Will create {(total_segments + batch_size - 1) // batch_size} intermediate videos")
        else:
            raise KeyError("'story_segments' not found in JSON")

    def split_text_into_chunks(self, text):
        """
        Split text into chunks that fit Kokoro's optimal length
        Tries to split at sentence boundaries
        """
        if len(text) <= self.max_chars_per_chunk:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back
            sentence = sentence + '.'
            
            # Check if adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) <= self.max_chars_per_chunk:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def generate_audio_kokoro(self, text, output_path):
        """
        Generate audio from text using Kokoro TTS
        Handles text splitting for longer passages
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            
        Returns:
            Path to generated audio file or None if audio disabled
        """
        if not self.include_audio or self.pipeline is None:
            return None

        try:
            # Split text into manageable chunks
            text_chunks = self.split_text_into_chunks(text)
            
            if len(text_chunks) > 1:
                print(f"  🎙️  Generating audio in {len(text_chunks)} chunks...", end=" ")
            else:
                print(f"  🎙️  Generating audio...", end=" ")
            
            audio_chunks = []
            
            # Generate audio for each chunk
            for i, chunk in enumerate(text_chunks):
                # Generate audio using Kokoro
                generator = self.pipeline(chunk, voice=self.voice)
                
                # Kokoro yields multiple segments, we need to concatenate them
                chunk_audio_segments = []
                for _, _, audio in generator:
                    chunk_audio_segments.append(audio)
                
                # Concatenate all segments for this chunk
                if chunk_audio_segments:
                    chunk_audio = np.concatenate(chunk_audio_segments)
                    audio_chunks.append(chunk_audio)
            
            # Concatenate all chunks with small pause between them (0.3 seconds)
            if audio_chunks:
                pause_samples = int(0.3 * self.sample_rate)
                pause = np.zeros(pause_samples)
                
                final_audio_parts = []
                for i, chunk_audio in enumerate(audio_chunks):
                    final_audio_parts.append(chunk_audio)
                    if i < len(audio_chunks) - 1:  # Don't add pause after last chunk
                        final_audio_parts.append(pause)
                
                final_audio = np.concatenate(final_audio_parts)
                
                # Save to file
                sf.write(output_path, final_audio, self.sample_rate)
                print("✓")
                return output_path
            else:
                print("⚠️  No audio generated")
                return None
                
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return None

    def find_panel_path(self, page_number, panel_id):
        """Find the correct panel file path"""
        page_folder = self.base_panels_folder / f"page_{page_number:03d}_panels"

        possible_names = [
            f"panel_{panel_id:02d}.jpg",
            f"panel_{panel_id:02d}.png",
            f"panel_{panel_id:02d}.jpeg",
        ]

        for name in possible_names:
            panel_path = page_folder / name
            if panel_path.exists():
                return panel_path

        # Search parent folder as fallback
        for name in possible_names:
            panel_path = self.base_panels_folder / name
            if panel_path.exists():
                return panel_path

        return None

    def create_panel_clip(self, page_number, panel_id, duration, is_fallback=False):
        """Create a video clip from a panel image"""
        panel_path = self.find_panel_path(page_number, panel_id)

        if not panel_path:
            raise FileNotFoundError(
                f"Panel not found: Page {page_number}, Panel {panel_id}"
            )

        if not is_fallback:
            self.last_used_panel = str(panel_path)

        clip = ImageClip(str(panel_path)).set_duration(duration)
        return clip

    def create_segment_video(self, segment):
        """Create video for a single story segment with Kokoro TTS"""
        segment_id = segment['segment_id']
        story_text = segment['story_paragraph']
        panels = segment['panels']
        page_number = segment['page_number']

        # Handle both old and new JSON formats
        if panels and isinstance(panels[0], int):
            panels = [{'page': page_number, 'panel_id': pid, 'reading_order': i+1}
                     for i, pid in enumerate(panels)]

        # Calculate durations
        if not self.include_audio:
            panel_duration = 3.0
            total_duration = panel_duration * max(len(panels), 1)
            audio_clip = None
        else:
            audio_path = os.path.join(self.temp_dir, f"segment_{segment_id}.wav")
            audio_file = self.generate_audio_kokoro(story_text, audio_path)
            
            if audio_file and os.path.exists(audio_file):
                try:
                    audio_clip = AudioFileClip(audio_file)
                    total_duration = audio_clip.duration
                    panel_duration = total_duration / max(len(panels), 1)
                except Exception as e:
                    print(f"  ⚠️  Error loading audio: {e}, using silent video")
                    panel_duration = 3.0
                    total_duration = panel_duration * max(len(panels), 1)
                    audio_clip = None
            else:
                # Fallback to silent video if audio generation fails
                print("  ⚠️  Using silent video for this segment")
                panel_duration = 3.0
                total_duration = panel_duration * max(len(panels), 1)
                audio_clip = None

        # Handle empty panels
        if not panels or len(panels) == 0:
            if self.last_used_panel is None:
                if audio_clip:
                    audio_clip.close()
                return None
            
            clip = ImageClip(self.last_used_panel).set_duration(total_duration)
            if audio_clip:
                clip = clip.set_audio(audio_clip)
            return clip

        # Create clips for each panel
        panel_clips = []
        for panel_ref in panels:
            page_num = panel_ref['page']
            panel_id = panel_ref['panel_id']

            try:
                img_clip = self.create_panel_clip(page_num, panel_id, panel_duration)
                panel_clips.append(img_clip)
            except FileNotFoundError as e:
                print(f"  ⚠ {e}")
                continue

        if not panel_clips:
            if self.last_used_panel:
                clip = ImageClip(self.last_used_panel).set_duration(total_duration)
                if audio_clip:
                    clip = clip.set_audio(audio_clip)
                return clip
            if audio_clip:
                audio_clip.close()
            return None

        # Concatenate panels
        video_clip = concatenate_videoclips(panel_clips, method="compose")

        # Add audio (must be added AFTER concatenation to avoid None references)
        if audio_clip:
            try:
                video_clip = video_clip.set_audio(audio_clip)
            except Exception as e:
                print(f"  ⚠️  Error setting audio: {e}")

        # Clean up panel clips (but NOT the audio clip yet - it's still referenced)
        for clip in panel_clips:
            clip.close()

        return video_clip

    def process_batch(self, segments, batch_num, resolution=(1280, 720)):
        """Process a batch of segments and save as intermediate video"""
        print(f"\n{'='*70}")
        print(f"PROCESSING BATCH {batch_num}")
        print(f"Segments: {len(segments)}")
        print(f"{'='*70}")

        segment_clips = []
        skipped = 0

        for i, segment in enumerate(segments, 1):
            segment_id = segment['segment_id']
            print(f"  [{i}/{len(segments)}] Segment {segment_id}...", end=" ")
            
            try:
                clip = self.create_segment_video(segment)
                if clip is not None:
                    segment_clips.append(clip)
                    print("✓")
                else:
                    skipped += 1
                    print("⚠ Skipped")
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
                skipped += 1
                continue

        if not segment_clips:
            print(f"  ⚠ No valid segments in batch {batch_num}")
            return None

        print(f"\n  Combining {len(segment_clips)} segments...")
        try:
            batch_video = concatenate_videoclips(segment_clips, method="compose")
        except Exception as e:
            print(f"  ✗ Error concatenating clips: {e}")
            # Clean up before raising
            for clip in segment_clips:
                try:
                    clip.close()
                except:
                    pass
            raise

        print(f"  Resizing to {resolution[0]}x{resolution[1]}...")
        batch_video = batch_video.resize(resolution)

        # Save batch video
        batch_path = os.path.join(self.batch_temp_dir, f"batch_{batch_num:03d}.mp4")
        print(f"  Writing batch video: {batch_path}")
        
        try:
            batch_video.write_videofile(
                batch_path,
                fps=24,
                codec='libx264',
                audio_codec='aac' if self.include_audio else None,
                temp_audiofile=os.path.join(self.temp_dir, f'temp-audio-batch-{batch_num}.m4a'),
                remove_temp=True,
                preset='medium',
                threads=4,
                verbose=True,  # Show progress bars
                logger='bar'   # Use progress bar logger
            )
        except Exception as e:
            print(f"  ✗ Error writing video: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clean up - close video and all segment clips
            try:
                batch_video.close()
            except:
                pass
            
            for clip in segment_clips:
                try:
                    clip.close()
                except:
                    pass
        
        gc.collect()

        print(f"  ✓ Batch {batch_num} complete: {batch_path}")
        return batch_path

    def merge_batch_videos(self, batch_paths, output_path):
        """Merge all batch videos into final output"""
        print(f"\n{'='*70}")
        print(f"MERGING {len(batch_paths)} BATCH VIDEOS")
        print(f"{'='*70}")

        batch_clips = []
        for i, batch_path in enumerate(batch_paths, 1):
            print(f"  [{i}/{len(batch_paths)}] Loading {os.path.basename(batch_path)}...")
            clip = VideoFileClip(batch_path)
            batch_clips.append(clip)

        print(f"\n  Concatenating {len(batch_clips)} videos...")
        final_video = concatenate_videoclips(batch_clips, method="compose")

        print(f"  Writing final video: {output_path}")
        final_video.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac' if self.include_audio else None,
            temp_audiofile=os.path.join(self.temp_dir, 'temp-audio-final.m4a'),
            remove_temp=True,
            preset='medium',
            threads=4,
            verbose=True,  # Show progress bars
            logger='bar'   # Use progress bar logger
        )

        # Clean up
        final_video.close()
        for clip in batch_clips:
            clip.close()
        
        gc.collect()

        print(f"\n  ✓ Final video complete!")
        print(f"     Location: {output_path}")
        print(f"     Duration: {final_video.duration:.1f}s")

    def generate_video(self, fps=24, resolution=(1280, 720)):
        """Generate the complete video with Kokoro TTS"""
        print("\n" + "="*70)
        print("VIDEO GENERATION WITH KOKORO TTS")
        print("="*70)
        print(f"JSON: {self.json_path}")
        print(f"Base panels folder: {self.base_panels_folder}")
        print(f"Output: {self.output_path}")
        print(f"Audio: {'Kokoro TTS' if self.include_audio else 'Disabled'}")
        print(f"Voice: {self.voice}")
        print(f"Language: {self.lang_code}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
        print(f"Batch size: {self.batch_size}")
        print("="*70)

        segments = self.data['story_segments']
        total_segments = len(segments)
        num_batches = (total_segments + self.batch_size - 1) // self.batch_size

        print(f"\nProcessing {total_segments} segments in {num_batches} batches...")

        # Process segments in batches
        batch_paths = []

        for batch_num in range(num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_segments)
            batch_segments = segments[start_idx:end_idx]

            print(f"\nBatch {batch_num + 1}/{num_batches}: "
                  f"Segments {start_idx + 1}-{end_idx}")

            batch_path = self.process_batch(batch_segments, batch_num + 1, resolution)

            if batch_path:
                batch_paths.append(batch_path)
            
            print(f"  🧹 Cleaning up memory...")
            gc.collect()

        if not batch_paths:
            raise ValueError("No valid batches created!")

        # Merge batches
        if len(batch_paths) == 1:
            print(f"\nOnly one batch created, moving to final output...")
            shutil.move(batch_paths[0], self.output_path)
        else:
            self.merge_batch_videos(batch_paths, self.output_path)

        # Final summary
        print(f"\n{'='*70}")
        print(f"✓ VIDEO CREATED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Location: {self.output_path}")
        print(f"Total segments: {total_segments}")
        print(f"Batches processed: {len(batch_paths)}")
        print(f"TTS Model: Kokoro (82M parameters)")
        print(f"Voice: {self.voice}")
        print(f"{'='*70}")

    def cleanup(self):
        """Clean up all temporary files"""
        print(f"\n🧹 Cleaning up temporary files...")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Also clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Cleanup complete")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # INSTALL DEPENDENCIES FIRST (run in Colab cell):
    # !pip install -q kokoro>=0.9.2 soundfile
    # !apt-get -qq -y install espeak-ng > /dev/null 2>&1
    
    JSON_PATH = "/content/story_output/panel_to_story_mapping.json"
    BASE_PANELS_FOLDER = "/content/knight_king_analysis"
    OUTPUT_VIDEO = "/content/knight_king_video_kokoro.mp4"
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     VIDEO GENERATOR WITH KOKORO TTS (OPEN-WEIGHT)           ║
║                                                              ║
║  ✓ Free & open-source TTS (82M parameters)                  ║
║  ✓ No API limits or quota restrictions                      ║
║  ✓ Multiple voices (American/British, Male/Female)          ║
║  ✓ Automatic text chunking for long passages                ║
║  ✓ Memory-efficient batch processing                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Available voices:
    # American Female: af_heart, af_sky, af_bella
    # American Male: am_adam, am_michael
    # British Female: bf_emma, bf_isabella
    # British Male: bm_george, bm_lewis
    
    VOICE = "af_bella"  # Choose your preferred voice
    LANG_CODE = "a"     # 'a' for American English, 'b' for British English
    BATCH_SIZE = 3    # Adjust based on RAM (6-24)
    
    generator = KokoroTTSVideoGenerator(
        json_path=JSON_PATH,
        base_panels_folder=BASE_PANELS_FOLDER,
        output_path=OUTPUT_VIDEO,
        include_audio=True,
        batch_size=BATCH_SIZE,
        voice=VOICE,
        lang_code=LANG_CODE
    )
    
    try:
        # Generate video with Kokoro TTS
        generator.generate_video(fps=24, resolution=(720, 1280))
    finally:
        # Always cleanup
        generator.cleanup()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    GENERATION COMPLETE!                      ║
╚══════════════════════════════════════════════════════════════╝

📝 NOTES:
  • Kokoro is completely free with no API limits
  • Text is automatically split into optimal chunks
  • 82M parameter model delivers high quality
  • Apache-licensed for any use case
  • Multiple voice options available
  • Adjust BATCH_SIZE if running out of RAM

🎙️ VOICE OPTIONS:
  Female American: af_heart (warm), af_sky (clear), af_bella (expressive)
  Male American: am_adam (deep), am_michael (smooth)
  Female British: bf_emma (elegant), bf_isabella (refined)
  Male British: bm_george (authoritative), bm_lewis (professional)
""")