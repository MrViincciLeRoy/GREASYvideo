"""
Memory-Efficient Video Generator with Kokoro TTS - OPTIMIZED WITH THREADING
Uses concurrent.futures for parallel processing of audio generation and video segments
"""

import json
import os
import gc
from pathlib import Path
import tempfile
import shutil
import soundfile as sf
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time

# Handle both old and new moviepy import structures
try:
    from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, VideoFileClip
except ImportError:
    try:
        from moviepy import ImageClip, concatenate_videoclips, AudioFileClip, VideoFileClip
    except ImportError:
        raise ImportError(
            "MoviePy import failed. Please install: pip install moviepy==1.0.3 imageio==2.31.1"
        )

class KokoroTTSVideoGenerator:
    def __init__(self, json_path, base_panels_folder, output_path="output_video.mp4", 
                 include_audio=True, batch_size=12, voice='af_heart', lang_code='a',
                 max_workers=4):
        """
        Initialize the video generator with Kokoro TTS and threading support
        
        Args:
            json_path: Path to panel_to_story_mapping.json
            base_panels_folder: Base folder containing page_XXX_panels subfolders
            output_path: Path for output video file
            include_audio: Whether to include audio narration
            batch_size: Number of segments to process before creating intermediate video
            voice: Kokoro voice to use
            lang_code: Language code ('a' for American English, 'b' for British English)
            max_workers: Number of parallel workers for threading (default: 4)
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
        self.max_workers = max_workers
        
        self.max_chars_per_chunk = 400

        # Initialize Kokoro pipeline
        if self.include_audio:
            print(f"🎙️  Loading Kokoro TTS (voice: {voice}, lang: {lang_code})...")
            print(f"🧵 Threading enabled with {max_workers} workers")
            try:
                from kokoro import KPipeline
                self.pipeline = KPipeline(lang_code=lang_code)
                print("✓ Kokoro TTS loaded successfully!")
            except Exception as e:
                print(f"⚠️  Warning: Could not load Kokoro TTS: {e}")
                print("    Falling back to silent video mode")
                self.include_audio = False
                self.pipeline = None
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
        """Split text into chunks that fit Kokoro's optimal length"""
        if len(text) <= self.max_chars_per_chunk:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence = sentence + '.'
            
            if len(current_chunk) + len(sentence) <= self.max_chars_per_chunk:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def generate_audio_kokoro(self, text, output_path):
        """Generate audio from text using Kokoro TTS"""
        if not self.include_audio or self.pipeline is None:
            return None

        try:
            text_chunks = self.split_text_into_chunks(text)
            
            if len(text_chunks) > 1:
                print(f"  🎙️  Generating audio in {len(text_chunks)} chunks...", end=" ")
            else:
                print(f"  🎙️  Generating audio...", end=" ")
            
            audio_chunks = []
            
            for i, chunk in enumerate(text_chunks):
                generator = self.pipeline(chunk, voice=self.voice)
                chunk_audio_segments = []
                for _, _, audio in generator:
                    chunk_audio_segments.append(audio)
                
                if chunk_audio_segments:
                    chunk_audio = np.concatenate(chunk_audio_segments)
                    audio_chunks.append(chunk_audio)
            
            if audio_chunks:
                pause_samples = int(0.3 * self.sample_rate)
                pause = np.zeros(pause_samples)
                
                final_audio_parts = []
                for i, chunk_audio in enumerate(audio_chunks):
                    final_audio_parts.append(chunk_audio)
                    if i < len(audio_chunks) - 1:
                        final_audio_parts.append(pause)
                
                final_audio = np.concatenate(final_audio_parts)
                sf.write(output_path, final_audio, self.sample_rate)
                print("✓")
                return output_path
            else:
                print("⚠️  No audio generated")
                return None
                
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return None

    def generate_audio_parallel(self, segments: List[Dict]) -> Dict[int, str]:
        """
        Generate audio for multiple segments in parallel using ThreadPoolExecutor
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Dictionary mapping segment_id to audio file path
        """
        audio_paths = {}
        
        if not self.include_audio:
            return audio_paths
        
        print(f"\n🎵 Generating audio for {len(segments)} segments in parallel...")
        start_time = time.time()
        
        def generate_single_audio(segment):
            segment_id = segment['segment_id']
            story_text = segment['story_paragraph']
            audio_path = os.path.join(self.temp_dir, f"segment_{segment_id}.wav")
            
            audio_file = self.generate_audio_kokoro(story_text, audio_path)
            return segment_id, audio_file
        
        # Use ThreadPoolExecutor for I/O-bound audio generation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(generate_single_audio, seg): seg for seg in segments}
            
            for future in as_completed(futures):
                try:
                    segment_id, audio_file = future.result()
                    if audio_file:
                        audio_paths[segment_id] = audio_file
                except Exception as e:
                    segment = futures[future]
                    print(f"  ❌ Error generating audio for segment {segment['segment_id']}: {e}")
        
        elapsed = time.time() - start_time
        print(f"✓ Generated {len(audio_paths)} audio files in {elapsed:.2f}s "
              f"({elapsed/len(segments):.2f}s per segment)")
        
        return audio_paths

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

    def create_segment_video(self, segment, audio_path=None):
        """
        Create video for a single story segment
        
        Args:
            segment: Segment dictionary
            audio_path: Pre-generated audio file path (if available)
        """
        segment_id = segment['segment_id']
        panels = segment['panels']
        page_number = segment['page_number']

        if panels and isinstance(panels[0], int):
            panels = [{'page': page_number, 'panel_id': pid, 'reading_order': i+1}
                     for i, pid in enumerate(panels)]

        # Load audio if available
        audio_clip = None
        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path)
                total_duration = audio_clip.duration
                panel_duration = total_duration / max(len(panels), 1)
            except Exception as e:
                print(f"  ⚠️  Error loading audio: {e}, using silent video")
                panel_duration = 3.0
                total_duration = panel_duration * max(len(panels), 1)
                audio_clip = None
        else:
            panel_duration = 3.0
            total_duration = panel_duration * max(len(panels), 1)

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

        # Add audio
        if audio_clip:
            try:
                video_clip = video_clip.set_audio(audio_clip)
            except Exception as e:
                print(f"  ⚠️  Error setting audio: {e}")

        # Clean up panel clips
        for clip in panel_clips:
            clip.close()

        return video_clip

    def create_segment_videos_parallel(self, segments: List[Dict], audio_paths: Dict[int, str]) -> List[Tuple[int, any]]:
        """
        Create video clips for multiple segments in parallel
        
        Args:
            segments: List of segment dictionaries
            audio_paths: Dictionary mapping segment_id to audio file path
            
        Returns:
            List of tuples (segment_id, video_clip)
        """
        print(f"\n🎬 Creating video clips for {len(segments)} segments in parallel...")
        start_time = time.time()
        
        segment_videos = []
        
        def create_single_video(segment):
            segment_id = segment['segment_id']
            audio_path = audio_paths.get(segment_id)
            
            try:
                clip = self.create_segment_video(segment, audio_path)
                return segment_id, clip, None
            except Exception as e:
                return segment_id, None, str(e)
        
        # Use ThreadPoolExecutor for video clip creation (I/O bound)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(create_single_video, seg): seg for seg in segments}
            
            for future in as_completed(futures):
                try:
                    segment_id, clip, error = future.result()
                    if clip is not None:
                        segment_videos.append((segment_id, clip))
                    elif error:
                        print(f"  ❌ Error creating video for segment {segment_id}: {error}")
                except Exception as e:
                    segment = futures[future]
                    print(f"  ❌ Exception for segment {segment['segment_id']}: {e}")
        
        # Sort by segment_id to maintain order
        segment_videos.sort(key=lambda x: x[0])
        
        elapsed = time.time() - start_time
        print(f"✓ Created {len(segment_videos)} video clips in {elapsed:.2f}s "
              f"({elapsed/len(segments):.2f}s per segment)")
        
        return segment_videos

    def process_batch(self, segments, batch_num, resolution=(1280, 720)):
        """Process a batch of segments with parallel audio and video generation"""
        print(f"\n{'='*70}")
        print(f"PROCESSING BATCH {batch_num} (PARALLEL MODE)")
        print(f"Segments: {len(segments)}")
        print(f"{'='*70}")

        # Step 1: Generate all audio files in parallel
        audio_paths = self.generate_audio_parallel(segments)

        # Step 2: Create all video clips in parallel
        segment_videos = self.create_segment_videos_parallel(segments, audio_paths)

        if not segment_videos:
            print(f"  ⚠ No valid segments in batch {batch_num}")
            return None

        # Extract just the clips
        segment_clips = [clip for _, clip in segment_videos]

        print(f"\n  Combining {len(segment_clips)} segments...")
        try:
            batch_video = concatenate_videoclips(segment_clips, method="compose")
        except Exception as e:
            print(f"  ✗ Error concatenating clips: {e}")
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
                verbose=False,
                logger=None
            )
        except Exception as e:
            print(f"  ✗ Error writing video: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
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
            verbose=False,
            logger=None
        )

        duration = final_video.duration

        final_video.close()
        for clip in batch_clips:
            clip.close()
        
        gc.collect()

        print(f"\n  ✓ Final video complete!")
        print(f"     Location: {output_path}")
        print(f"     Duration: {duration:.1f}s")

    def generate_video(self, fps=24, resolution=(1280, 720)):
        """Generate the complete video with parallel processing"""
        print("\n" + "="*70)
        print("VIDEO GENERATION WITH KOKORO TTS (PARALLEL MODE)")
        print("="*70)
        print(f"JSON: {self.json_path}")
        print(f"Base panels folder: {self.base_panels_folder}")
        print(f"Output: {self.output_path}")
        print(f"Audio: {'Kokoro TTS' if self.include_audio else 'Disabled'}")
        print(f"Voice: {self.voice}")
        print(f"Language: {self.lang_code}")
        print(f"Resolution: {resolution[0]}x{resolution[1]}")
        print(f"Batch size: {self.batch_size}")
        print(f"Parallel workers: {self.max_workers}")
        print("="*70)

        segments = self.data['story_segments']
        total_segments = len(segments)
        num_batches = (total_segments + self.batch_size - 1) // self.batch_size

        print(f"\nProcessing {total_segments} segments in {num_batches} batches...")

        batch_paths = []
        overall_start = time.time()

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

        total_time = time.time() - overall_start

        # Final summary
        print(f"\n{'='*70}")
        print(f"✓ VIDEO CREATED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Location: {self.output_path}")
        print(f"Total segments: {total_segments}")
        print(f"Batches processed: {len(batch_paths)}")
        print(f"TTS Model: Kokoro (82M parameters)")
        print(f"Voice: {self.voice}")
        print(f"Total time: {total_time/60:.2f} minutes ({total_time/total_segments:.2f}s per segment)")
        print(f"Parallel workers: {self.max_workers}")
        print(f"{'='*70}")

    def cleanup(self):
        """Clean up all temporary files"""
        print(f"\n🧹 Cleaning up temporary files...")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Cleanup complete")