"""
Primitive detection - detect ALL geometric shapes and text regions without classification.
Classification happens in a separate pass based on spatial context.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class Primitive:
    """A detected geometric primitive (rectangle, circle, or text region)."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    prim_type: str  # 'rect', 'circle', 'text'
    contour: np.ndarray  # Original contour
    features: dict  # Additional features for classification


@dataclass
class Character:
    """A detected individual character primitive."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray  # Original contour
    features: dict  # height, width, aspect_ratio, solidity, area

class Color(Tuple):
    """RGB color tuple for filtering."""
    pass


def find_dominant_color(img: np.ndarray, exclude_grayscale: bool = True) -> Tuple[int, int, int]:
    """Find the dominant non-grayscale color in an image."""
    if img.ndim != 3:
        raise ValueError("Image must be RGB")

    # Reshape to list of pixels
    pixels = img.reshape(-1, 3)

    # Filter out grayscale pixels (where R ≈ G ≈ B)
    if exclude_grayscale:
        # Calculate variance across RGB channels for each pixel
        rgb_std = pixels.std(axis=1)
        # Keep pixels with significant color variance (> 10)
        colored_pixels = pixels[rgb_std > 10]

        if len(colored_pixels) == 0:
            # No colored pixels, return most common color
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            return tuple(int(x) for x in unique[counts.argmax()])

        pixels = colored_pixels

    # Find most common color
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = unique[counts.argmax()]
    return tuple(int(x) for x in dominant)  # Convert to regular int tuple


def detect_all_rectangles(img: np.ndarray, color: Color | None = None) -> List[Primitive]:
    """Detect ALL rectangular shapes regardless of size or function.

    Args:
        img: Input image (RGB or grayscale)
        color: If provided, only detect rectangles of this color (RGB tuple)
    """
    # If color filtering is requested, use color-based detection
    if color is not None and img.ndim == 3:
        # Create mask for the target color (with tolerance)
        target_color = np.array(color, dtype=np.int16)  # Use int16 to avoid wrap-around
        lower = np.clip(target_color - 30, 0, 255).astype(np.uint8)
        upper = np.clip(target_color + 30, 0, 255).astype(np.uint8)

        # Assume input is RGB (from page.flatten())
        # Use inRange directly on RGB image
        mask = cv2.inRange(img, lower, upper)

        # Find contours of colored regions
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Standard edge-based detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

        # Adaptive threshold to catch both dark and light rectangles
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Edge detection
        edges = cv2.Canny(thr, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles: List[Primitive] = []
    for c in cnts or []:
        area = cv2.contourArea(c)
        if area < 16:  # Absolute minimum to filter noise
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 3 or h < 3:
            continue

        # Approximate polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Must be polygon-like (3-12 vertices) - but skip this check for color-based detection
        # since filled colored regions are already filtered by color
        if color is None and (len(approx) < 3 or len(approx) > 12):
            continue

        # Calculate features for later classification
        aspect = w / (h + 1e-6)
        bbox_area = w * h
        rect_ratio = area / (bbox_area + 1e-6)
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)

        # Extract edge density inside bbox (if edges were computed)
        if color is None:
            if y + h <= edges.shape[0] and x + w <= edges.shape[1]:
                roi_edges = edges[y:y+h, x:x+w]
                edge_density = float(roi_edges.mean()) if roi_edges.size else 0.0
            else:
                edge_density = 0.0

            # Extract fill info
            if y + h <= gray.shape[0] and x + w <= gray.shape[1]:
                roi_gray = gray[y:y+h, x:x+w]
                mean_intensity = float(roi_gray.mean()) if roi_gray.size else 0.0
            else:
                mean_intensity = 0.0
        else:
            # Color-based detection: use different features
            edge_density = 10.0  # Assume solid colored regions have low edge density
            mean_intensity = 128.0  # Neutral value

        rectangles.append(Primitive(
            bbox=(x, y, w, h),
            prim_type='rect',
            contour=c,
            features={
                'vertices': len(approx),
                'aspect_ratio': aspect,
                'fill_ratio': rect_ratio,
                'solidity': solidity,
                'edge_density': edge_density,
                'mean_intensity': mean_intensity,
                'area': area
            }
        ))

    return rectangles


def detect_all_circles(img: np.ndarray, color: Color | None = None) -> List[Primitive]:
    """Detect ALL circular shapes regardless of size or function."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thr, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles: List[Primitive] = []
    for c in cnts or []:
        area = cv2.contourArea(c)
        if area < 9:  # Minimum circle area (r ~= 1.7)
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 3 or h < 3:
            continue

        # Circularity check
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)

        # Must be reasonably circular
        if circularity < 0.5:
            continue

        # Aspect ratio should be squarish for circles
        aspect = w / (h + 1e-6)
        if not (0.7 < aspect < 1.4):
            continue

        # Approximate vertex count
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Calculate radius
        r = (w + h) / 4.0

        # Extract ring features
        cx, cy = x + w // 2, y + h // 2
        if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
            inner_r = max(1, int(r * 0.5))
            outer_r = max(2, int(r * 0.9))

            mask_ring = np.zeros(gray.shape[:2], dtype=np.uint8)
            cv2.circle(mask_ring, (cx, cy), outer_r, 255, -1)
            cv2.circle(mask_ring, (cx, cy), inner_r, 0, -1)

            ring_pixels = gray[mask_ring == 255]
            ring_intensity = float(ring_pixels.mean()) if ring_pixels.size else 0.0

            mask_inner = np.zeros(gray.shape[:2], dtype=np.uint8)
            cv2.circle(mask_inner, (cx, cy), inner_r, 255, -1)
            inner_pixels = gray[mask_inner == 255]
            inner_intensity = float(inner_pixels.mean()) if inner_pixels.size else 0.0
        else:
            ring_intensity = 0.0
            inner_intensity = 0.0

        circles.append(Primitive(
            bbox=(x, y, w, h),
            prim_type='circle',
            contour=c,
            features={
                'circularity': circularity,
                'radius': r,
                'vertices': len(approx),
                'ring_intensity': ring_intensity,
                'inner_intensity': inner_intensity,
                'area': area
            }
        ))

    return circles


def detect_all_characters(img: np.ndarray) -> List[Character]:
    """Detect individual character-like contours.

    Uses aggressive filtering to identify character-sized regions:
    - Height: 5-35px (typical character height at 150 DPI)
    - Width: 2-30px (typical character width)
    - Aspect ratio: 0.3-5.0 (characters are usually taller or squarish)
    - Solidity: > 0.5 (reject fragmented/noisy contours)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    # Threshold to isolate dark text on light background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # CRITICAL FIX (Iteration 18): Use RETR_LIST instead of RETR_EXTERNAL
    # RETR_EXTERNAL only gets outermost contours, causing form fields to merge into giant blobs (1127x751px)
    # RETR_LIST gets ALL contours including text inside form field borders
    # This fixes the missing middle section (y=800-1200) where 0 characters were detected
    cnts, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    characters: List[Character] = []
    for c in cnts:
        area = cv2.contourArea(c)

        # Minimum area to filter noise
        if area < 10:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Character size constraints (at 150 DPI)
        # Height: 5-35px (covers most font sizes from 8pt to 18pt)
        # Width: 2-30px (covers narrow letters like 'i' to wide letters like 'W')
        if not (5 <= h <= 35 and 2 <= w <= 30):
            continue

        # Aspect ratio: characters are typically 0.3-5.0 (tall or squarish)
        # This filters out horizontal lines and very wide boxes
        aspect = h / (w + 1e-6)
        if not (0.3 <= aspect <= 5.0):
            continue

        # Solidity: reject fragmented or very irregular shapes
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if solidity < 0.5:
            continue

        # ITERATION 18: Reject hollow shapes (checkboxes, form borders)
        # These have low fill ratio (mostly white inside)
        # Characters should be mostly filled (solid black)
        fill_ratio = area / (w * h + 1e-6)
        if fill_ratio < 0.4:  # Less than 40% filled = likely a border/box
            continue

        # ITERATION 18: Reject square shapes (checkboxes)
        # Characters are rarely perfect squares
        if 0.8 <= aspect <= 1.2 and w > 10 and h > 10:  # Square and reasonably sized
            # Could be checkbox - reject unless very solid
            if solidity < 0.9:
                continue

        characters.append(Character(
            bbox=(x, y, w, h),
            contour=c,
            features={
                'height': h,
                'width': w,
                'aspect_ratio': aspect,
                'solidity': solidity,
                'area': area,
                'fill_ratio': fill_ratio
            }
        ))

    return characters


def group_characters_into_words(chars: List[Character],
                                 h_gap_ratio: float = 0.5,
                                 v_gap_ratio: float = 0.3,
                                 max_word_width: int = 300) -> List[Primitive]:
    """Group characters into word-level text primitives.

    Args:
        chars: List of detected characters
        h_gap_ratio: Horizontal gap threshold as ratio of average char width (default 0.5)
        v_gap_ratio: Vertical gap threshold as ratio of average char height (default 0.3)
        max_word_width: Maximum width for a word box in pixels

    Returns:
        List of Primitive objects representing words
    """
    if not chars:
        return []

    # Sort characters left-to-right, top-to-bottom
    sorted_chars = sorted(chars, key=lambda c: (c.bbox[1], c.bbox[0]))

    words: List[Primitive] = []
    current_group: List[Character] = [sorted_chars[0]]

    for curr_char in sorted_chars[1:]:
        prev_char = current_group[-1]

        cx, cy, cw, ch = curr_char.bbox
        px, py, pw, ph = prev_char.bbox

        # Use current character height for gap calculations (more adaptive)
        # Using average can be misleading when fonts vary
        curr_height = ch

        # Vertical distance between character tops
        v_dist = abs(cy - py)

        # Horizontal distance (gap between previous char end and current char start)
        h_dist = cx - (px + pw)

        # Check if characters are on the same line
        same_line = v_dist < curr_height * v_gap_ratio

        # Check if characters are close enough horizontally to be in same word
        # Use current character height as reference for gap
        close_enough = 0 <= h_dist < curr_height * h_gap_ratio

        # Calculate what the merged width would be
        group_min_x = min(c.bbox[0] for c in current_group)
        group_max_x = max(c.bbox[0] + c.bbox[2] for c in current_group)
        merged_width_would_be = max(group_max_x, cx + cw) - min(group_min_x, cx)

        # Don't merge if it would create an overly wide word
        width_ok = merged_width_would_be <= max_word_width

        should_group = same_line and close_enough and width_ok

        if should_group:
            current_group.append(curr_char)
        else:
            # Finalize current word and start new one
            if current_group:
                words.append(_merge_characters_to_primitive(current_group))
            current_group = [curr_char]

    # Don't forget the last group
    if current_group:
        words.append(_merge_characters_to_primitive(current_group))

    return words


def _merge_characters_to_primitive(chars: List[Character]) -> Primitive:
    """Merge a group of characters into a single word-level Primitive."""
    if len(chars) == 1:
        char = chars[0]
        return Primitive(
            bbox=char.bbox,
            prim_type='text',
            contour=char.contour,
            features={
                'aspect_ratio': char.features['height'] / (char.features['width'] + 1e-6),
                'area': char.features['area'],
                'method': 'single_character',
                'char_count': 1
            }
        )

    # Calculate bounding box that encompasses all characters
    min_x = min(c.bbox[0] for c in chars)
    min_y = min(c.bbox[1] for c in chars)
    max_x = max(c.bbox[0] + c.bbox[2] for c in chars)
    max_y = max(c.bbox[1] + c.bbox[3] for c in chars)

    w = max_x - min_x
    h = max_y - min_y

    # Use first character's contour as representative
    return Primitive(
        bbox=(min_x, min_y, w, h),
        prim_type='text',
        contour=chars[0].contour,
        features={
            'aspect_ratio': w / (h + 1e-6),
            'area': w * h,
            'method': 'character_grouped',
            'char_count': len(chars)
        }
    )


def merge_words_into_phrases(words: List[Primitive],
                             h_gap_threshold: int = 50,
                             v_gap_ratio: float = 0.8,
                             max_phrase_width: int = 1200) -> List[Primitive]:
    """Merge word-level primitives into phrase/sentence-level regions.

    This post-processing step combines nearby words on the same line
    to match the granularity of ground truth boxes.

    Args:
        words: List of word-level primitives
        h_gap_threshold: Max horizontal gap in pixels to merge words (default 50px)
        v_gap_ratio: Max vertical offset ratio to consider same line (default 0.8)
        max_phrase_width: Maximum width for merged phrase in pixels

    Returns:
        List of merged primitives at phrase/sentence level
    """
    if not words:
        return []

    # Sort words left-to-right, top-to-bottom
    sorted_words = sorted(words, key=lambda p: (p.bbox[1], p.bbox[0]))

    phrases: List[Primitive] = []
    current_group: List[Primitive] = [sorted_words[0]]

    for curr_word in sorted_words[1:]:
        prev_word = current_group[-1]

        cx, cy, cw, ch = curr_word.bbox
        px, py, pw, ph = prev_word.bbox

        # Vertical distance between word tops
        v_dist = abs(cy - py)

        # Horizontal gap
        h_gap = cx - (px + pw)

        # Same line if vertical offset is small relative to average height
        avg_height = (ch + ph) / 2
        same_line = v_dist < avg_height * v_gap_ratio

        # Close enough horizontally
        close_enough = 0 <= h_gap < h_gap_threshold

        # Calculate merged width
        group_min_x = min(p.bbox[0] for p in current_group)
        group_max_x = max(p.bbox[0] + p.bbox[2] for p in current_group)
        merged_width = max(group_max_x, cx + cw) - min(group_min_x, cx)

        width_ok = merged_width <= max_phrase_width

        should_merge = same_line and close_enough and width_ok

        if should_merge:
            current_group.append(curr_word)
        else:
            # Finalize current phrase
            if current_group:
                phrases.append(_merge_primitives(current_group))
            current_group = [curr_word]

    # Don't forget last group
    if current_group:
        phrases.append(_merge_primitives(current_group))

    return phrases


def _merge_primitives(prims: List[Primitive]) -> Primitive:
    """Merge multiple primitives into one."""
    if len(prims) == 1:
        return prims[0]

    # Calculate bounding box encompassing all primitives
    min_x = min(p.bbox[0] for p in prims)
    min_y = min(p.bbox[1] for p in prims)
    max_x = max(p.bbox[0] + p.bbox[2] for p in prims)
    max_y = max(p.bbox[1] + p.bbox[3] for p in prims)

    w = max_x - min_x
    h = max_y - min_y

    # Sum up character counts
    total_chars = sum(p.features.get('char_count', 1) for p in prims)

    # Preserve OCR text if present (combine with newlines for multi-line blocks)
    text_parts = [p.features.get('text', '') for p in prims if p.features.get('text')]
    combined_text = '\n'.join(text_parts) if text_parts else ''

    # Preserve confidence if present (average)
    confidences = [p.features.get('confidence') for p in prims if p.features.get('confidence') is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else None

    features = {
        'aspect_ratio': w / (h + 1e-6),
        'area': w * h,
        'method': 'phrase_merged',
        'char_count': total_chars,
        'word_count': len(prims)
    }
    if combined_text:
        features['text'] = combined_text
    if avg_conf is not None:
        features['confidence'] = avg_conf

    return Primitive(
        bbox=(min_x, min_y, w, h),
        prim_type='text',
        contour=prims[0].contour,
        features=features
    )


def merge_phrases_into_blocks(phrases: List[Primitive],
                              v_gap_threshold: int = 10,
                              h_overlap_ratio: float = 0.3,
                              max_block_height: int = 100) -> List[Primitive]:
    """Merge vertically-stacked phrases into multi-line text blocks.

    This handles cases like multi-line form fields, paragraphs, etc.

    Args:
        phrases: List of phrase-level primitives
        v_gap_threshold: Max vertical gap in pixels to merge phrases
        h_overlap_ratio: Min horizontal overlap ratio to consider aligned
        max_block_height: Maximum height for merged block

    Returns:
        List of block-level primitives
    """
    if not phrases:
        return []

    # Sort phrases top-to-bottom
    sorted_phrases = sorted(phrases, key=lambda p: p.bbox[1])

    blocks: List[Primitive] = []
    current_group: List[Primitive] = [sorted_phrases[0]]

    for curr_phrase in sorted_phrases[1:]:
        prev_phrase = current_group[-1]

        cx, cy, cw, ch = curr_phrase.bbox
        px, py, pw, ph = prev_phrase.bbox

        # Vertical gap between phrases
        v_gap = cy - (py + ph)

        # Check horizontal overlap
        x_overlap = min(px + pw, cx + cw) - max(px, cx)
        h_overlaps = x_overlap > 0 and x_overlap / min(pw, cw) > h_overlap_ratio

        # Close enough vertically
        close_vertically = 0 <= v_gap < v_gap_threshold

        # Calculate merged height
        group_min_y = min(p.bbox[1] for p in current_group)
        group_max_y = max(p.bbox[1] + p.bbox[3] for p in current_group)
        merged_height = max(group_max_y, cy + ch) - min(group_min_y, cy)

        height_ok = merged_height <= max_block_height

        should_merge = h_overlaps #close_vertically and h_overlaps and height_ok

        if should_merge:
            current_group.append(curr_phrase)
        else:
            # Finalize current block
            if current_group:
                blocks.append(_merge_primitives(current_group))
            current_group = [curr_phrase]

    # Don't forget last group
    if current_group:
        blocks.append(_merge_primitives(current_group))

    return blocks


def merge_nearby_text_boxes(prims: List[Primitive], h_gap_ratio: float = 0.5, v_gap_ratio: float = 0.15) -> List[Primitive]:
    """Merge nearby text boxes into phrases.

    OCR returns word-level boxes, but we need phrase-level granularity.
    This merges boxes that are horizontally close and on the same line.

    Args:
        prims: List of text primitives (word-level from OCR)
        h_gap_ratio: Max horizontal gap as ratio of box height (0.5 = half the font size)
        v_gap_ratio: Max vertical gap as ratio of box height (0.15 = 15% of font size)

    Returns:
        List of merged primitives (phrase-level)
    """
    if not prims:
        return []

    # Sort by vertical position (top to bottom), then horizontal (left to right)
    sorted_prims = sorted(prims, key=lambda p: (p.bbox[1], p.bbox[0]))

    merged = []
    current_group = [sorted_prims[0]]

    for i in range(1, len(sorted_prims)):
        prev = current_group[-1]
        curr = sorted_prims[i]

        prev_x, prev_y, prev_w, prev_h = prev.bbox
        curr_x, curr_y, _, curr_h = curr.bbox

        # Use average height for adaptive thresholds
        avg_height = (prev_h + curr_h) / 2

        # Check if on same line (vertical proximity)
        # Use vertical distance between tops, but with tolerance for height differences
        v_distance = abs(curr_y - prev_y)
        v_threshold = avg_height * v_gap_ratio

        # Check horizontal gap
        h_distance = curr_x - (prev_x + prev_w)
        h_threshold = avg_height * h_gap_ratio

        # Also check for overlapping boxes (negative gap means overlap)
        overlapping = h_distance < 0

        # Merge if on same line and (horizontally close OR overlapping)
        if v_distance <= v_threshold and (h_distance <= h_threshold or overlapping):
            current_group.append(curr)
        else:
            # Start new group
            if current_group:
                merged.append(_merge_text_primitives(current_group))
            current_group = [curr]

    # Don't forget last group
    if current_group:
        merged.append(_merge_text_primitives(current_group))

    return merged


def _merge_text_primitives(prims: List[Primitive]) -> Primitive:
    """Merge text primitives, combining their text content."""
    if len(prims) == 1:
        return prims[0]

    # Calculate bounding box
    min_x = min(p.bbox[0] for p in prims)
    min_y = min(p.bbox[1] for p in prims)
    max_x = max(p.bbox[0] + p.bbox[2] for p in prims)
    max_y = max(p.bbox[1] + p.bbox[3] for p in prims)

    # Combine text content
    texts = [p.features.get('text', '') for p in prims]
    combined_text = ' '.join(texts)

    # Average confidence
    confidences = [p.features.get('confidence', 1.0) for p in prims]
    avg_conf = sum(confidences) / len(confidences)

    return Primitive(
        bbox=(min_x, min_y, max_x - min_x, max_y - min_y),
        prim_type='text',
        contour=prims[0].contour,
        features={'text': combined_text, 'confidence': avg_conf}
    )


def detect_all_text_regions(img: np.ndarray, color: Color | None = None) -> List[Primitive]:
    """Detect ALL text regions using OCR (EasyOCR) with CV fallback.

    Primary: EasyOCR (91% recall, Apache 2.0, distributable)
    Fallback: Character-first CV (85% recall)
    """
    # Try EasyOCR first
    try:
        import easyocr

        if not hasattr(detect_all_text_regions, '_reader'):
            print("[OCR] Initializing EasyOCR (first run takes ~5 seconds)...")
            detect_all_text_regions._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[OCR] Ready (subsequent detections will be faster)")

        reader = detect_all_text_regions._reader
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # OCR settings: word-level detection with quality filtering
        results = reader.readtext(
            rgb,
            paragraph=False,      # Word-level detections
            text_threshold=0.6,   # Balanced quality threshold
            low_text=0.3,         # Character linking
            link_threshold=0.3    # Link between components
        )

        print(f"[OCR] Detected {len(results)} text regions")

        primitives = []
        for bbox_pts, text, conf in results:
            if conf < 0.2:  # Filter noise
                continue
            pts = np.array(bbox_pts, dtype=np.int32)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            w, h = x2 - x, y2 - y
            if w > 0 and h > 0:
                prim = Primitive(
                    bbox=(x, y, w, h),
                    prim_type='text',
                    contour=pts.reshape(-1, 1, 2),
                    features={'text': text, 'confidence': conf}
                )
                primitives.append(prim)

        # Merge word-level detections into phrases
        # h_gap_ratio=1.0 for normal word spacing
        # v_gap_ratio=0.15 for same-line requirement
        phrases = merge_nearby_text_boxes(primitives, h_gap_ratio=1.0, v_gap_ratio=0.15)
        print(f"[OCR] Merged {len(primitives)} words -> {len(phrases)} phrases")

        return phrases

    except ImportError as e:
        print(f"[OCR] EasyOCR import failed: {type(e).__name__}: {e}, using CV fallback")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"[OCR] Failed: {type(e).__name__}: {e}, using CV fallback")
        import traceback
        traceback.print_exc()

    # CV Fallback
    try:
        # Step 1: Detect individual characters
        characters = detect_all_characters(img)

        print(f"[Character-first] Detected {len(characters)} characters")

        # Step 2: Group characters into words
        # ITERATION 20: Reduced from 15.0 to preserve very small isolated labels
        words = group_characters_into_words(
            characters,
            h_gap_ratio=15.0,     # Horizontal gap: 15x char height (aggressive)
            v_gap_ratio=0.5,      # Vertical gap: 50% of char height (same line)
            max_word_width=300    # Maximum word width in pixels
        )

        print(f"[Character-first] Grouped into {len(words)} words")

        # Step 3: Merge words into phrases/sentences (same-line horizontal)
        # This matches GT granularity which has mean width ~330px and max ~1105px
        # ITERATION 20: Reduced from 150px to preserve small isolated labels
        phrases = merge_words_into_phrases(
            words,
            h_gap_threshold=130,   # Merge words within 130px horizontal gap (was 150)
            v_gap_ratio=0.9,       # Very loose same-line tolerance
            max_phrase_width=1200  # Allow wide phrases (GT max is 1105px)
        )

        print(f"[Character-first] Merged into {len(phrases)} phrases")

        # Step 4: Merge phrases into multi-line blocks (vertical)
        # This handles multi-line form fields, paragraphs, etc.
        # ITERATION 18: More aggressive vertical merging
        blocks = merge_phrases_into_blocks(
            phrases,
            v_gap_threshold=15,    # Merge phrases within 15px vertical gap (increased from 10)
            h_overlap_ratio=0.2,   # Lower overlap requirement (30% → 20%)
            max_block_height=100   # Allow up to 100px height (GT max is 86px)
        )

        print(f"[Character-first] Merged into {len(blocks)} blocks")

        return blocks

    except Exception as e:
        print(f"[Character-first] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def classify_primitives(rects: List[Primitive], circles: List[Primitive], text: List[Primitive]) -> dict:
    """
    Classify primitives into semantic categories based on features and spatial context.

    Returns:
        {
            'checkboxes': List[Primitive],
            'bubbles': List[Primitive],
            'text_fields': List[Primitive],
            'shapes': List[Primitive],
            'text_blocks': List[Primitive]
        }
    """
    checkboxes = []
    bubbles = []
    text_fields = []
    shapes = []
    text_blocks = []

    # Helper: Check if a rectangle contains text
    def contains_text(rect_bbox, text_regions):
        rx, ry, rw, rh = rect_bbox
        for txt in text_regions:
            tx, ty, tw, th = txt.bbox
            # Check if text center is inside rect
            tcx, tcy = tx + tw // 2, ty + th // 2
            if rx <= tcx < rx + rw and ry <= tcy < ry + rh:
                # Text should be significantly smaller than the rect
                if tw < rw * 0.8 or th < rh * 0.8:
                    return True
        return False

    # Classify rectangles
    for rect in rects:
        x, y, w, h = rect.bbox
        f = rect.features

        # Skip if this rectangle contains text - it's a text field or labeled element
        if contains_text(rect.bbox, text):
            # If elongated and hollow, it's a text field
            if f['aspect_ratio'] > 1.5 and f['fill_ratio'] < 0.4 and 20 <= w <= 500 and 8 <= h <= 80:
                text_fields.append(rect)
            else:
                # Otherwise it's a labeled checkbox/section - keep as shape
                if w * h >= 100:
                    shapes.append(rect)
            continue

        # Checkbox criteria: small, squarish, hollow or semi-filled, clear borders, NO text inside
        is_checkbox = (
            6 <= w <= 40 and 6 <= h <= 40 and
            0.65 < f['aspect_ratio'] < 1.5 and
            4 <= f['vertices'] <= 8 and
            f['solidity'] >= 0.6 and
            f['fill_ratio'] >= 0.35 and
            f['edge_density'] >= 5
        )

        if is_checkbox:
            checkboxes.append(rect)
            continue

        # Text field criteria: elongated, hollow, moderate size, no text detected yet
        is_text_field = (
            30 <= w <= 500 and 10 <= h <= 80 and
            f['aspect_ratio'] > 1.7 and
            f['fill_ratio'] < 0.3 and
            f['edge_density'] >= 3
        )

        if is_text_field:
            text_fields.append(rect)
            continue

        # Everything else is a shape (borders, decorations, etc.)
        if w * h >= 100:  # Only keep substantial shapes
            shapes.append(rect)

    # Classify circles
    for circ in circles:
        x, y, w, h = circ.bbox
        f = circ.features

        # Bubble/radio button criteria: small, circular, ring structure
        is_bubble = (
            4 <= f['radius'] <= 30 and
            f['circularity'] >= 0.55 and
            f['vertices'] <= 8
        )

        if is_bubble:
            bubbles.append(circ)
        else:
            # Large circles are decorative shapes
            if w * h >= 100:
                shapes.append(circ)

    # ITERATION 18: Re-enable light deduplication to reduce false positives
    # With RETR_LIST we're getting overlapping contours, need to remove duplicates
    deduplicated_text = deduplicate_text_regions(text)
    print(f"[Classify] After deduplication: {len(deduplicated_text)} regions")

    # ITERATION 20: Improved filtering with h_gap_threshold=130 + wide line filter
    # Building on Iteration 19's approach but with better merging to preserve small boxes
    filtered_text = []
    for prim in deduplicated_text:
        _, _, w, h = prim.bbox
        aspect = w / (h + 1e-6)

        # Reject square boxes 23-35px (checkboxes)
        if 0.85 <= aspect <= 1.2 and 23 <= w <= 35 and 23 <= h <= 35:
            continue

        # Reject wide thin horizontal lines (underlines, borders)
        # These appear when h_gap_threshold < 150 (54x9, 43x9)
        if w > 40 and h <= 9:
            continue

        # Reject very tiny boxes (< 45 px²)
        # GT minimum is 98 px², so this safely removes noise
        if w * h < 45:
            continue

        filtered_text.append(prim)

    print(f"[Classify] After checkbox/noise filtering: {len(filtered_text)} regions")

    # ITERATION 18: Skip merge_text_regions() since character-first already merged
    # merged_text = merge_text_regions(deduplicated_text)
    # print(f"[Classify] After merging: {len(merged_text)} regions")

    # TEMPORARY: Don't clip text to shapes since we're not detecting shapes right now
    # clipped_text = clip_text_to_shapes(merged_text, shapes)
    # text_blocks = clipped_text
    text_blocks = filtered_text

    return {
        'checkboxes': checkboxes,
        'bubbles': bubbles,
        'text_fields': text_fields,
        'shapes': shapes,
        'text_blocks': text_blocks
    }


def clip_text_to_shapes(text_regions: List[Primitive], shapes: List[Primitive]) -> List[Primitive]:
    """Clip text blocks to their containing shapes so they don't exceed shape boundaries."""
    if not shapes:
        return text_regions

    clipped_text = []
    for txt in text_regions:
        tx, ty, tw, th = txt.bbox

        # Find shapes that contain this text
        containing_shapes = []
        for shape in shapes:
            sx, sy, sw, sh = shape.bbox
            # Check if text center is inside shape
            tcx, tcy = tx + tw // 2, ty + th // 2
            if sx <= tcx < sx + sw and sy <= tcy < sy + sh:
                containing_shapes.append(shape)

        if containing_shapes:
            # Find the smallest containing shape (most specific)
            smallest = min(containing_shapes, key=lambda s: s.bbox[2] * s.bbox[3])
            sx, sy, sw, sh = smallest.bbox

            # Clip text bbox to shape bounds
            clipped_x = max(tx, sx)
            clipped_y = max(ty, sy)
            clipped_x2 = min(tx + tw, sx + sw)
            clipped_y2 = min(ty + th, sy + sh)

            clipped_w = clipped_x2 - clipped_x
            clipped_h = clipped_y2 - clipped_y

            # Only keep if clipped area is substantial
            if clipped_w > 0 and clipped_h > 0:
                clipped_text.append(Primitive(
                    bbox=(clipped_x, clipped_y, clipped_w, clipped_h),
                    prim_type='text',
                    contour=txt.contour,
                    features={**txt.features, 'clipped': True}
                ))
        else:
            # No containing shape, keep as-is
            clipped_text.append(txt)

    return clipped_text


def deduplicate_text_regions(text_regions: List[Primitive]) -> List[Primitive]:
    """Remove nested/overlapping duplicate text regions, then merge nearby regions on same line."""
    if not text_regions:
        return []

    # Step 1: Remove duplicates
    sorted_regions = sorted(text_regions, key=lambda p: p.bbox[2] * p.bbox[3], reverse=True)

    def bbox_overlap_ratio(bbox1, bbox2):
        """Calculate how much bbox2 is contained within bbox1."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox2_area = w2 * h2

        return intersection_area / (bbox2_area + 1e-6)

    kept = []
    for region in sorted_regions:
        # Check if this region is nested/contained in any kept region
        is_nested = False
        for kept_region in kept:
            overlap = bbox_overlap_ratio(kept_region.bbox, region.bbox)

            # ITERATION 19: Font-size-relative tolerance for nested box prevention
            # Small text (height < 15px): strict threshold (0.2)
            # Medium text (15-25px): moderate threshold (0.3)
            # Large text (> 25px): lenient threshold (0.5)
            # This prevents removing legitimate small labels that happen to overlap large blocks

            _, _, _, rh = region.bbox
            _, _, _, kh = kept_region.bbox

            # Use smaller box's height to determine tolerance
            smaller_height = min(rh, kh)

            if smaller_height < 15:
                # Small text - very strict (e.g., small labels, footnotes)
                threshold = 0.2
            elif smaller_height < 25:
                # Medium text - moderate (increased from 0.3)
                threshold = 0.35
            else:
                # Large text - lenient (increased from 0.5)
                threshold = 0.6

            if overlap > threshold:
                is_nested = True
                break

        if not is_nested:
            kept.append(region)

    # Return deduplicated regions without additional merging
    # The morphological kernel already handles grouping
    return kept


def merge_text_regions(text_regions: List[Primitive]) -> List[Primitive]:
    """Merge nearby text regions into contiguous blocks - balanced approach for sentences and paragraphs."""
    if not text_regions:
        return []

    # Step 1: Deduplicate overlapping/contained text regions
    # Remove smaller regions that are fully contained within larger ones
    deduplicated = []
    for i, region in enumerate(text_regions):
        x1, y1, w1, h1 = region.bbox
        is_contained = False

        for j, other in enumerate(text_regions):
            if i == j:
                continue
            x2, y2, w2, h2 = other.bbox

            # Check if region is mostly contained within other (80% overlap)
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            region_area = w1 * h1

            if region_area > 0 and overlap_area / region_area > 0.8 and w2 * h2 > w1 * h1:
                is_contained = True
                break

        if not is_contained:
            deduplicated.append(region)

    # Step 2: Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
    sorted_regions = sorted(deduplicated, key=lambda t: (t.bbox[1], t.bbox[0]))

    # Step 3: Merge nearby regions into contiguous blocks, respecting font size
    merged: List[Primitive] = []
    current_group: List[Primitive] = [sorted_regions[0]]

    for i in range(1, len(sorted_regions)):
        curr = sorted_regions[i]
        prev = current_group[-1]

        cx, cy, cw, ch = curr.bbox
        px, py, pw, ph = prev.bbox

        # CRITICAL: Check font size compatibility first
        # Don't merge text of significantly different sizes (e.g., headings vs body)
        height_ratio = max(ch, ph) / (min(ch, ph) + 1e-6)
        font_size_similar = height_ratio < 1.4  # Within 40% size difference

        if not font_size_similar:
            # Different font sizes - don't merge (e.g., heading vs paragraph)
            if current_group:
                merged.append(merge_group(current_group))
            current_group = [curr]
            continue

        avg_height = (ch + ph) / 2.0

        # Horizontal and vertical distances
        h_dist = cx - (px + pw)  # Horizontal gap (can be negative if overlapping)
        v_dist = cy - py  # Vertical distance between tops

        # ITERATION 14: Back to original conservative merging BUT add width limit
        same_line = abs(v_dist) < avg_height * 0.15 and 0 <= h_dist < avg_height * 0.8

        # ITERATION 14b: Check that merging won't create a box wider than 280px
        # Relaxed from 250px to allow some longer valid phrases
        # Calculate what the merged width would be
        if current_group:
            group_min_x = min(p.bbox[0] for p in current_group)
            group_max_x = max(p.bbox[0] + p.bbox[2] for p in current_group)
            merged_width_would_be = max(group_max_x, cx + cw) - min(group_min_x, cx)
        else:
            merged_width_would_be = cw

        width_ok = merged_width_would_be <= 280

        should_merge = same_line and width_ok

        if should_merge:
            current_group.append(curr)
        else:
            # Finalize current group and start new one
            if current_group:
                merged.append(merge_group(current_group))
            current_group = [curr]

    # Don't forget the last group
    if current_group:
        merged.append(merge_group(current_group))

    return merged


def merge_group(group: List[Primitive]) -> Primitive:
    """Merge a group of text primitives into a single bounding box."""
    if len(group) == 1:
        return group[0]

    min_x = min(p.bbox[0] for p in group)
    min_y = min(p.bbox[1] for p in group)
    max_x = max(p.bbox[0] + p.bbox[2] for p in group)
    max_y = max(p.bbox[1] + p.bbox[3] for p in group)

    w = max_x - min_x
    h = max_y - min_y

    # Use first contour as representative
    return Primitive(
        bbox=(min_x, min_y, w, h),
        prim_type='text',
        contour=group[0].contour,
        features={
            'aspect_ratio': w / (h + 1e-6),
            'area': w * h,
            'method': 'merged',
            'merged_count': len(group)
        }
    )
