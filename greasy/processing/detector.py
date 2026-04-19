
# panel_detector.py - IMPROVED VERSION

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field, asdict
import pytesseract
import os

@dataclass
class TextElement:
    """Individual text element (dialogue, SFX, etc.)"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    text_type: str = "unknown"  # 'dialogue', 'narration', 'sfx', 'caption'

@dataclass
class Panel:
    panel_id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    center_x: int
    center_y: int
    text: str = ""  # ALL text combined
    text_elements: List[TextElement] = field(default_factory=list)
    is_narration: bool = False
    confidence: float = 0.0
    reading_order: int = 0

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def dialogue_text(self) -> str:
        """Get only dialogue text"""
        return ' '.join([te.text for te in self.text_elements if te.text_type == 'dialogue'])

    @property
    def narration_text(self) -> str:
        """Get only narration text"""
        return ' '.join([te.text for te in self.text_elements if te.text_type == 'narration'])

    @property
    def sfx_text(self) -> str:
        """Get only sound effects text"""
        return ' '.join([te.text for te in self.text_elements if te.text_type == 'sfx'])

    def to_dict(self) -> Dict:
        """Convert panel to dictionary for JSON serialization."""
        return {
            'panel_id': self.panel_id,
            'position': {'x': self.x, 'y': self.y},
            'dimensions': {'width': self.width, 'height': self.height},
            'area': self.area,
            'center': {'x': self.center_x, 'y': self.center_y},
            'text': self.text,
            'text_elements': [
                {
                    'text': te.text,
                    'position': {'x': te.x, 'y': te.y},
                    'dimensions': {'width': te.width, 'height': te.height},
                    'confidence': te.confidence,
                    'type': te.text_type
                } for te in self.text_elements
            ],
            'dialogue_text': self.dialogue_text,
            'narration_text': self.narration_text,
            'sfx_text': self.sfx_text,
            'has_narration_box': self.is_narration,
            'ocr_confidence': self.confidence,
            'reading_order': self.reading_order
        }

@dataclass
class NarrationBox:
    narration_id: int
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float
    panel_id: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert narration box to dictionary."""
        return {
            'narration_id': self.narration_id,
            'position': {'x': self.x, 'y': self.y},
            'dimensions': {'width': self.width, 'height': self.height},
            'text': self.text,
            'confidence': self.confidence,
            'parent_panel_id': self.panel_id
        }

class PanelDetector:
    def __init__(self,
                 min_area_ratio: float = 0.03,
                 max_area_ratio: float = 0.95,
                 min_panel_dimension: int = 150,
                 comic_format: str = "auto"):  # NEW: auto, traditional, webtoon
        """
        Initialize panel detector with improved filtering.

        Args:
            min_area_ratio: Minimum panel area as fraction of page (default 3%)
            max_area_ratio: Maximum panel area as fraction of page
            min_panel_dimension: Minimum width or height in pixels (filters out small boxes)
            comic_format: "auto" (detect), "traditional" (rows), or "webtoon" (vertical strip)
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_panel_dimension = min_panel_dimension
        self.comic_format = comic_format

    def detect_panels(self, image_path: str, extract_text: bool = False,
                     detect_narration: bool = False) -> Tuple[List[Panel], List[NarrationBox]]:
        """
        Detect FULL panels with improved filtering to exclude speech bubbles and incomplete panels.
        Automatically detects comic format (traditional vs webtoon).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"\n{'='*70}")
        print(f"DETECTING PANELS: {image_path}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")

        # Detect format if set to auto
        if self.comic_format == "auto":
            detected_format = self._detect_comic_format(img.shape)
            print(f"Format detected: {detected_format}")
        else:
            detected_format = self.comic_format
            print(f"Format: {detected_format} (manual)")

        print(f"{'='*70}")

        strategies = [
            self._detect_adaptive_threshold,
            self._detect_edges,
            self._detect_morphological,
            self._detect_lines
        ]

        results = []
        for i, strategy in enumerate(strategies, 1):
            try:
                panels = strategy(img)
                # Apply improved filtering
                panels = self._filter_panels_advanced(panels, img.shape)

                if 2 <= len(panels) <= 15:
                    results.append((len(panels), panels, strategy.__name__))
                    print(f"  Strategy {i} ({strategy.__name__}): {len(panels)} panels")
            except Exception as e:
                print(f"  Strategy {i} failed: {e}")
                continue

        if not results:
            print("  ⚠ No valid results, using fallback")
            try:
                panels = strategies[0](img)
                panels = self._filter_panels_advanced(panels, img.shape)
            except:
                panels = []
        else:
            # Choose best result
            best = max(results, key=lambda x: x[0])
            panels = best[1]
            print(f"  ✓ Selected: {best[2]} with {len(panels)} panels")

        # Sort panels based on detected format
        panels = self._sort_panels_by_format(panels, detected_format)

        # Assign panel IDs and reading order
        for i, panel in enumerate(panels):
            panel.panel_id = i + 1
            panel.reading_order = i + 1

        print(f"\n  Final panel count: {len(panels)}")
        for panel in panels:
            print(f"    Panel {panel.panel_id}: {panel.width}x{panel.height} @ ({panel.x}, {panel.y})")

        narration_boxes = []

        # Detect narration boxes separately
        if detect_narration:
            narration_boxes = self.detect_narration_boxes(img, panels)
            print(f"  Found {len(narration_boxes)} narration boxes")

        # Extract text from panels
        if extract_text:
            panels = self.extract_all_text_from_panels(img, panels, narration_boxes)

        return panels, narration_boxes

    def _filter_panels_advanced(self, panels: List[Panel], img_shape: Tuple[int, int, int]) -> List[Panel]:
        """
        Advanced filtering to remove:
        - Speech bubbles (round/oval, small)
        - Sound effect boxes (spiky borders like "OHH!!!")
        - Incomplete panels (wrong aspect ratio, too thin)
        - Overlapping duplicates
        """
        page_height, page_width = img_shape[:2]
        page_area = page_height * page_width

        filtered = []

        for panel in panels:
            # Calculate properties
            aspect_ratio = panel.width / panel.height if panel.height > 0 else 0
            area_ratio = panel.area / page_area

            # FILTER 1: Size requirements
            # Panels must be substantial - both dimensions should be meaningful
            if panel.width < self.min_panel_dimension or panel.height < self.min_panel_dimension:
                continue

            # FILTER 2: Area requirements
            # Must be between min and max area ratios
            if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                continue

            # FILTER 3: Aspect ratio requirements
            # Panels should be rectangular-ish, not extremely thin/tall
            # Speech bubbles and SFX boxes often have extreme ratios
            if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                continue

            # FILTER 4: Absolute area check
            # Text elements (bubbles, SFX) are typically smaller than this
            min_absolute_area = 40000  # pixels - increased from 20000
            if panel.area < min_absolute_area:
                continue

            # FILTER 5: Text element detection
            # If it looks like a speech bubble or SFX box, skip it
            # SFX boxes (like "OHH!!!") are typically:
            # - Smaller than panels (but larger than regular bubbles)
            # - Have spiky/irregular borders
            # - Mostly white background with large text

            # Medium-small size with certain characteristics = text element
            if area_ratio < 0.05:  # Less than 5% of page
                # Additional checks for text elements

                # Check if width is much greater than height (caption-like)
                if aspect_ratio > 4.0:
                    continue  # Likely a caption or thin SFX box

                # Check if height is much greater than width (vertical text)
                if aspect_ratio < 0.25:
                    continue  # Likely vertical text element

                # If small AND square-ish, might be SFX box
                if 0.7 < aspect_ratio < 1.4 and area_ratio < 0.03:
                    continue  # Likely a square SFX box like "OHH!!!"

            # FILTER 6: Check if mostly on page edge (incomplete panel detection)
            edge_threshold = 10  # pixels from edge
            on_left_edge = panel.x < edge_threshold
            on_right_edge = (panel.x + panel.width) > (page_width - edge_threshold)
            on_top_edge = panel.y < edge_threshold
            on_bottom_edge = (panel.y + panel.height) > (page_height - edge_threshold)

            # If touching 3+ edges, might be incomplete
            edges_touched = sum([on_left_edge, on_right_edge, on_top_edge, on_bottom_edge])
            if edges_touched >= 3:
                continue

            filtered.append(panel)

        # Remove overlaps (keep larger panels)
        filtered = self._remove_overlaps_by_size(filtered)

        return filtered

    def _remove_overlaps_by_size(self, panels: List[Panel]) -> List[Panel]:
        """
        Remove overlapping panels, keeping larger ones.
        Improved to handle partial overlaps better.
        """
        if len(panels) <= 1:
            return panels

        # Sort by area (largest first)
        panels = sorted(panels, key=lambda p: p.area, reverse=True)
        kept = []

        for panel in panels:
            overlapped = False

            for kept_panel in kept:
                # Calculate overlap
                x1 = max(panel.x, kept_panel.x)
                y1 = max(panel.y, kept_panel.y)
                x2 = min(panel.x + panel.width, kept_panel.x + kept_panel.width)
                y2 = min(panel.y + panel.height, kept_panel.y + kept_panel.height)

                if x2 > x1 and y2 > y1:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    panel_overlap_ratio = overlap_area / panel.area
                    kept_overlap_ratio = overlap_area / kept_panel.area

                    # If significant overlap (>40% of either panel), consider it a duplicate
                    if panel_overlap_ratio > 0.4 or kept_overlap_ratio > 0.4:
                        overlapped = True
                        break

            if not overlapped:
                kept.append(panel)

        return kept

    def _is_text_element_not_panel(self, x: int, y: int, w: int, h: int,
                                    area: int, page_area: int, contour: np.ndarray,
                                    img: np.ndarray = None) -> bool:
        """
        Detect if a contour is a text element (speech bubble, SFX box) rather than a panel.

        Text elements include:
        - Speech bubbles (round/oval with dialogue)
        - Sound effect boxes (spiky borders like "OHH!!!")
        - Small caption boxes

        All of these should NOT be detected as panels.
        """
        # SIZE CHECK: Text elements are much smaller than panels
        # Speech bubbles and SFX boxes are typically < 3% of page
        area_ratio = area / page_area
        if area_ratio > 0.03:  # Larger than 3% might be a panel
            # But still check other factors
            pass
        elif area_ratio < 0.002:  # Too small to be anything useful
            return True

        # ABSOLUTE SIZE: Text elements usually smaller than 50,000 pixels
        if area < 50000 and area_ratio < 0.025:
            return True

        # DIMENSION CHECK: Very thin/tall elements are likely text boxes
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 8 or aspect_ratio < 0.12:
            return True

        # SHAPE ANALYSIS: Check border complexity
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            # Compactness: low values = irregular/spiky (SFX boxes)
            compactness = (4 * np.pi * area) / (perimeter * perimeter)

            # SFX boxes and speech bubbles have irregular borders (low compactness)
            # But they're also small, so combine with size
            if compactness < 0.3 and area_ratio < 0.03:
                return True  # Likely a spiky SFX box

            # Very circular = dialogue bubble
            if compactness > 0.85 and area_ratio < 0.02:
                return True

        # CHECK POSITION: If mostly white/empty with small text area, likely bubble
        if img is not None:
            roi = img[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Check if mostly white (background of speech bubbles)
            white_ratio = np.sum(gray_roi > 240) / gray_roi.size
            if white_ratio > 0.85 and area_ratio < 0.03:
                return True  # Likely a speech bubble with white background

        return False

    def detect_narration_boxes(self, img: np.ndarray, panels: List[Panel]) -> List[NarrationBox]:
        """
        Detect narration boxes - rectangular text boxes with borders.
        IMPROVED: Better distinction from speech bubbles and panel borders.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        narration_boxes = []
        page_area = img.shape[0] * img.shape[1]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Narration boxes are:
            # - Wide and rectangular (high aspect ratio)
            # - Medium-small size (not as big as panels)
            # - Have clear borders

            # SIZE: Between 0.5% and 10% of page
            if not (0.005 * page_area < area < 0.10 * page_area):
                continue

            # SHAPE: Wide rectangles (aspect ratio 2.5 to 12)
            if not (2.5 < aspect_ratio < 12):
                continue

            # Check if it's rectangular (has 4-ish corners)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) >= 4:
                # Extract text to verify it's actually a text box
                roi = gray[y:y+h, x:x+w]
                roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)

                try:
                    data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)

                    texts = []
                    confidences = []
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 30:
                            text = data['text'][i].strip()
                            if text:
                                texts.append(text)
                                confidences.append(int(conf))

                    # Must have text to be a narration box
                    if texts:
                        text_content = ' '.join(texts)
                        avg_conf = sum(confidences) / len(confidences)
                        panel_id = self._find_parent_panel(x, y, w, h, panels)

                        narration_box = NarrationBox(
                            narration_id=len(narration_boxes) + 1,
                            x=x, y=y, width=w, height=h,
                            text=text_content,
                            confidence=avg_conf,
                            panel_id=panel_id
                        )
                        narration_boxes.append(narration_box)
                except:
                    continue

        narration_boxes = self._remove_narration_overlaps(narration_boxes)

        # Mark panels that contain narration boxes
        for nb in narration_boxes:
            if nb.panel_id is not None:
                for panel in panels:
                    if panel.panel_id == nb.panel_id:
                        panel.is_narration = True
                        break

        return narration_boxes

    def extract_all_text_from_panels(self, img: np.ndarray, panels: List[Panel],
                                     narration_boxes: List[NarrationBox]) -> List[Panel]:
        """Extract ALL text from each panel using OCR."""
        for panel in panels:
            x1, y1, x2, y2 = panel.coords
            panel_img = img[y1:y2, x1:x2]

            # Preprocess for better OCR
            gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            try:
                data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

                all_text = []
                text_elements = []

                panel_narration_boxes = [nb for nb in narration_boxes if nb.panel_id == panel.panel_id]

                for i, conf in enumerate(data['conf']):
                    if int(conf) > 0:
                        text = data['text'][i].strip()
                        if text:
                            txt_x = data['left'][i]
                            txt_y = data['top'][i]
                            txt_w = data['width'][i]
                            txt_h = data['height'][i]

                            text_type = self._classify_text_type(
                                txt_x, txt_y, txt_w, txt_h,
                                text, panel_narration_boxes,
                                panel.width, panel.height
                            )

                            text_element = TextElement(
                                text=text,
                                x=panel.x + txt_x,
                                y=panel.y + txt_y,
                                width=txt_w,
                                height=txt_h,
                                confidence=float(conf),
                                text_type=text_type
                            )

                            text_elements.append(text_element)
                            all_text.append(text)

                panel.text = ' '.join(all_text)
                panel.text_elements = text_elements
                panel.confidence = sum([te.confidence for te in text_elements]) / len(text_elements) if text_elements else 0.0

            except Exception as e:
                print(f"Warning: OCR failed for panel {panel.panel_id}: {e}")
                panel.text = ""
                panel.text_elements = []
                panel.confidence = 0.0

        return panels

    def _classify_text_type(self, x: int, y: int, w: int, h: int, text: str,
                           narration_boxes: List[NarrationBox],
                           panel_width: int, panel_height: int) -> str:
        """Classify text as dialogue, narration, SFX, or caption."""
        # Check if inside a narration box
        for nb in narration_boxes:
            if (nb.x <= x <= nb.x + nb.width and
                nb.y <= y <= nb.y + nb.height):
                return 'narration'

        # Check if it looks like sound effects
        if text.isupper() and len(text) > 2:
            has_special = any(c in text for c in ['!', '*', '#', '@'])
            if has_special or w > panel_width * 0.3:
                return 'sfx'

        # Check if in caption area
        if y < panel_height * 0.15 or y > panel_height * 0.85:
            return 'caption'

        return 'dialogue'

    def _find_parent_panel(self, x: int, y: int, w: int, h: int,
                          panels: List[Panel]) -> Optional[int]:
        """Find which panel contains this narration box."""
        center_x = x + w // 2
        center_y = y + h // 2

        for panel in panels:
            if (panel.x <= center_x <= panel.x + panel.width and
                panel.y <= center_y <= panel.y + panel.height):
                return panel.panel_id
        return None

    def _remove_narration_overlaps(self, boxes: List[NarrationBox]) -> List[NarrationBox]:
        """Remove overlapping narration boxes."""
        if len(boxes) <= 1:
            return boxes

        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        kept = []

        for box in boxes:
            overlapped = False
            for kept_box in kept:
                x1 = max(box.x, kept_box.x)
                y1 = max(box.y, kept_box.y)
                x2 = min(box.x + box.width, kept_box.x + kept_box.width)
                y2 = min(box.y + box.height, kept_box.y + kept_box.height)

                if x2 > x1 and y2 > y1:
                    overlap_area = (x2 - x1) * (y2 - y1)
                    box_area = box.width * box.height
                    kept_area = kept_box.width * kept_box.height

                    if overlap_area / min(box_area, kept_area) > 0.3:
                        overlapped = True
                        break

            if not overlapped:
                kept.append(box)

        for i, box in enumerate(kept):
            box.narration_id = i + 1

        return kept

    def extract_panel_images(self, image_path: str, panels: List[Panel],
                            output_dir: str) -> Dict[int, str]:
        """Extract FULL panel images."""
        os.makedirs(output_dir, exist_ok=True)
        img = Image.open(image_path)
        panel_paths = {}

        for panel in panels:
            cropped = img.crop(panel.coords)
            filename = f"panel_{panel.panel_id:02d}.jpg"
            path = os.path.join(output_dir, filename)
            cropped.save(path, "JPEG", quality=95)
            panel_paths[panel.panel_id] = path

            print(f"  ✓ Saved panel {panel.panel_id} → {filename}")
            if panel.is_narration:
                print(f"    (includes narration box)")

        return panel_paths

    def _detect_adaptive_threshold(self, img: np.ndarray) -> List[Panel]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return self._extract_panels_from_binary(binary, img.shape)

    def _detect_edges(self, img: np.ndarray) -> List[Panel]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        return self._extract_panels_from_binary(edges, img.shape)

    def _detect_morphological(self, img: np.ndarray) -> List[Panel]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return self._extract_panels_from_binary(binary, img.shape)

    def _detect_lines(self, img: np.ndarray) -> List[Panel]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return self._detect_adaptive_threshold(img)

        line_mask = np.zeros_like(gray)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        kernel = np.ones((5, 5), np.uint8)
        line_mask = cv2.dilate(line_mask, kernel, iterations=2)
        return self._extract_panels_from_binary(line_mask, img.shape)

    def _extract_panels_from_binary(self, binary: np.ndarray, img_shape: Tuple[int, int, int]) -> List[Panel]:
        """Extract panels from binary image - basic extraction, filtering happens later."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        page_area = img_shape[0] * img_shape[1]
        panels = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Very basic filtering here - real filtering in _filter_panels_advanced
            if w > 50 and h > 50:  # Minimum absolute size
                # Skip obvious text elements (speech bubbles, SFX boxes)
                # This is a quick pre-filter before the more detailed filtering
                if area < page_area * 0.004:  # Less than 0.4% of page
                    continue  # Too small to be a panel

                panels.append(Panel(
                    panel_id=0,
                    x=x, y=y, width=w, height=h, area=area,
                    center_x=x + w // 2, center_y=y + h // 2
                ))

        # Don't sort here - let detect_panels handle it based on format
        return panels

    def _detect_comic_format(self, img_shape: Tuple[int, int, int]) -> str:
        """
        Detect if this is a traditional comic page or a webtoon/vertical strip.

        Args:
            img_shape: Image shape (height, width, channels)

        Returns:
            "traditional" or "webtoon"
        """
        height, width = img_shape[:2]
        aspect_ratio = height / width

        # Webtoons are typically very tall and narrow
        # Traditional comics are more square or wider
        if aspect_ratio > 3.0:  # More than 3x taller than wide
            return "webtoon"
        elif aspect_ratio > 2.0:  # 2-3x taller - likely webtoon
            # Additional check: webtoons usually have width around 800-1000px
            if 600 <= width <= 1200:
                return "webtoon"
            return "traditional"
        else:
            return "traditional"

    def _sort_panels_by_format(self, panels: List[Panel], format_type: str) -> List[Panel]:
        """
        Sort panels based on comic format.

        Args:
            panels: List of detected panels
            format_type: "traditional" or "webtoon"

        Returns:
            Sorted list of panels
        """
        if format_type == "webtoon":
            return self._sort_panels_webtoon(panels)
        else:
            return self._sort_panels_traditional(panels)

    def _sort_panels_webtoon(self, panels: List[Panel]) -> List[Panel]:
        """
        Sort panels for webtoon format (vertical strip, top-to-bottom only).

        In webtoons, panels are stacked vertically with no side-by-side layout.
        Simply sort by Y position (top to bottom).
        """
        if not panels:
            return []

        print(f"  📱 Sorting as WEBTOON (vertical strip)")

        # Sort by Y position only (top to bottom)
        sorted_panels = sorted(panels, key=lambda p: p.y)

        # Assign reading order
        for i, panel in enumerate(sorted_panels, 1):
            panel.reading_order = i

        return sorted_panels

    def _sort_panels_traditional(self, panels: List[Panel]) -> List[Panel]:
        """
        Sort panels for traditional comic format (rows with left-to-right flow).
        Groups panels into rows, then sorts left-to-right within each row.
        """
        if not panels:
            return []

        print(f"  📖 Sorting as TRADITIONAL (rows)")

        # Sort by Y position first
        panels = sorted(panels, key=lambda p: p.y)

        # Group into rows based on vertical overlap
        rows = []
        current_row = [panels[0]]

        for panel in panels[1:]:
            # Check if this panel overlaps vertically with current row
            # Two panels are in same row if their Y ranges overlap significantly

            # Calculate vertical overlap
            y1_start = min(p.y for p in current_row)
            y1_end = max(p.y + p.height for p in current_row)
            y2_start = panel.y
            y2_end = panel.y + panel.height

            overlap_start = max(y1_start, y2_start)
            overlap_end = min(y1_end, y2_end)
            overlap = max(0, overlap_end - overlap_start)

            # If panels overlap by at least 40% of the smaller panel's height, same row
            min_height = min(panel.height, min(p.height for p in current_row))
            if overlap > min_height * 0.4:
                current_row.append(panel)
            else:
                # Start new row
                rows.append(sorted(current_row, key=lambda x: x.x))
                current_row = [panel]

        # Don't forget last row
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x.x))

        # Flatten rows into final reading order
        sorted_panels = [p for row in rows for p in row]

        # Assign reading order
        for i, panel in enumerate(sorted_panels, 1):
            panel.reading_order = i

        return sorted_panels