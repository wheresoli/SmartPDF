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


def detect_all_text_regions(img: np.ndarray, color: Color | None = None) -> List[Primitive]:
    """Detect ALL text regions using multiple methods."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    text_regions: List[Primitive] = []

    # MSER DISABLED - too slow and creates character-level detections that get merged anyway
    # We only use morphological detection for word-level detection which is much faster

    # Method 2: Morphological text detection for connected text regions
    try:
        # Threshold to get text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Balanced horizontal kernel: connect letters into words without over-fragmenting
        # Ground truth shows word-level boxes (83x15, 179x15, etc.), not merged phrases
        # Kernel (6,1): balance between merging letter fragments and avoiding phrase merging
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
        h_dilate = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel, iterations=1)

        # NO vertical dilation - text blocks are single-line
        combined = h_dilate

        # Find connected text regions
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)

            # Handle massive merged regions by subdividing them
            if h > 80 or w > 1000:
                # This is likely multiple lines/regions merged together
                # Extract the ROI and find individual text lines within it
                roi = combined[y:y+h, x:x+w]

                # Find horizontal projections to split into lines
                h_proj = roi.sum(axis=1)  # Sum across width
                h_proj = h_proj / 255  # Normalize to number of white pixels

                # Find runs of text (where projection has significant content)
                # Use threshold of 5% of width to filter noise
                threshold = w * 0.05
                in_text = h_proj > threshold
                changes = np.diff(np.concatenate([[0], in_text.astype(int), [0]]))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]

                # Create bbox for each text line
                for line_start, line_end in zip(starts, ends):
                    line_h = line_end - line_start
                    if line_h < 5:  # Too thin
                        continue

                    # Extract this line and find horizontal segments
                    line_roi = roi[line_start:line_end, :]
                    v_proj = line_roi.sum(axis=0) / 255

                    # Find horizontal segments (words/phrases separated by gaps)
                    # Use threshold to identify significant text vs noise
                    h_threshold = line_h * 0.5  # At least 50% of line height has text
                    in_segment = v_proj > h_threshold
                    h_changes = np.diff(np.concatenate([[0], in_segment.astype(int), [0]]))
                    seg_starts = np.where(h_changes == 1)[0]
                    seg_ends = np.where(h_changes == -1)[0]

                    if len(seg_starts) == 0:
                        continue

                    # Merge nearby segments (within 20px gap) to avoid over-fragmentation
                    merged_segs = []
                    if len(seg_starts) > 0:
                        current_start = seg_starts[0]
                        current_end = seg_ends[0]

                        for i in range(1, len(seg_starts)):
                            gap = seg_starts[i] - current_end
                            if gap < 20:  # Merge if gap is small
                                current_end = seg_ends[i]
                            else:
                                merged_segs.append((current_start, current_end))
                                current_start = seg_starts[i]
                                current_end = seg_ends[i]

                        merged_segs.append((current_start, current_end))

                    for seg_start, seg_end in merged_segs:
                        seg_w = seg_end - seg_start
                        if seg_w < 10:  # Too narrow
                            continue
                        if seg_w * line_h < 50:  # Too small
                            continue

                        seg_x = x + seg_start
                        seg_y = y + line_start

                        aspect = seg_w / (line_h + 1e-6)
                        if aspect < 0.3:
                            continue

                        text_regions.append(Primitive(
                            bbox=(seg_x, seg_y, seg_w, line_h),
                            prim_type='text',
                            contour=c,
                            features={
                                'aspect_ratio': aspect,
                                'area': seg_w * line_h,
                                'method': 'morphology_subdivided'
                            }
                        ))
                continue

            # Normal-sized regions: apply standard filters
            if w < 10 or h < 5:
                continue
            if w * h < 50:
                continue

            # Text should be somewhat horizontal
            aspect = w / (h + 1e-6)
            if aspect < 0.3:
                continue

            text_regions.append(Primitive(
                bbox=(x, y, w, h),
                prim_type='text',
                contour=c,
                features={
                    'aspect_ratio': aspect,
                    'area': w * h,
                    'method': 'morphology'
                }
            ))
    except Exception as e:
        print(f"Morphological text detection failed: {e}")

    return text_regions


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

    # Deduplicate overlapping/nested regions first
    deduplicated_text = deduplicate_text_regions(text)

    # Then merge nearby boxes on the same line to reduce false positives
    merged_text = merge_text_regions(deduplicated_text)

    # TEMPORARY: Don't clip text to shapes since we're not detecting shapes right now
    # clipped_text = clip_text_to_shapes(merged_text, shapes)
    # text_blocks = clipped_text
    text_blocks = merged_text

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
        # Check if this region is mostly contained in any kept region
        is_duplicate = False
        for kept_region in kept:
            overlap = bbox_overlap_ratio(kept_region.bbox, region.bbox)
            if overlap > 0.8:  # 80% contained
                is_duplicate = True
                break

        if not is_duplicate:
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

        # ULTRA CONSERVATIVE: Only merge if directly adjacent (within one character width)
        # This creates individual word-level detections matching the heuristic
        # Y-coords must be nearly identical (within 15% of height) and horizontal gap tiny (< 0.8x char width)
        same_line = abs(v_dist) < avg_height * 0.15 and 0 <= h_dist < avg_height * 0.8

        # DISABLED: Multi-line paragraph merging to keep text boxes small and precise
        # # Condition 2: Multi-line paragraph - merge lines that are vertically stacked
        # # Bottom of prev is close to top of curr, and they start at similar x positions
        # line_spacing = cy - (py + ph)  # Gap between bottom of prev and top of curr
        # vertically_stacked = (
        #     0 <= line_spacing < avg_height * 1.5 and  # Normal line spacing (0 to 1.5x height)
        #     abs(cx - px) < avg_height * 3.0  # Left-aligned or similar indentation
        # )

        should_merge = same_line  # Only merge same-line words

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
