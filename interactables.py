from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid
import cv2

class Z:
    TOP = "top"
    BOTTOM = "bottom"

class Alignment(Z):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    MIDDLE = "middle"

ALIGNMENT: list[str] = [Alignment.LEFT, Alignment.CENTER, Alignment.RIGHT, Alignment.TOP, Alignment.MIDDLE, Alignment.BOTTOM]
CIRCLE: str = "circle"
SQUARE: str = "square"
RECTANGLE: str = "rectangle"
ROUNDED_RECTANGLE: str = "rounded-rectangle"
POLYGON: str = "polygon"
LINE: str = "line"

@dataclass
class Interactable:
    id: str
    kind: str  # e.g. "highlight", "censor", "checkbox", "bubble"
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixel coords
    meta: Dict[str, Any]
    z: int = 0 # Height order for overlapping interactables
    primitive: Any = None  # Optional reference to source Primitive (for OCR text, features, etc.)

    # Use dataclass-generated __init__; bbox should be passed directly

    def to_dict(self) -> Dict[str, Any]:
        def _json_safe(v):
            try:
                import numpy as _np
            except Exception:
                _np = None
            # Convert numpy scalars and dtypes to native Python types
            if _np is not None:
                if isinstance(v, (_np.integer,)):
                    return int(v)
                if isinstance(v, (_np.floating,)):
                    return float(v)
                if isinstance(v, (_np.bool_,)):
                    return bool(v)
                if isinstance(v, _np.ndarray):
                    return v.tolist()
            if isinstance(v, (list, tuple)):
                return type(v)([_json_safe(x) for x in v])
            if isinstance(v, dict):
                return {str(k): _json_safe(val) for k, val in v.items()}
            return v

        d = asdict(self)
        # Ensure bbox and meta are JSON-safe
        d["bbox"] = _json_safe(d.get("bbox"))
        d["meta"] = _json_safe(d.get("meta"))
        # Remove primitive field (contains non-serializable numpy arrays)
        d.pop("primitive", None)
        return d

    def on_click(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "meta": self.meta}

    @classmethod
    def detect_all(cls, img: np.ndarray) -> List[Interactable]:
        """Attempt to identify generic interactables; returns all matches.
        For each candidate subregion, call `detect(roi)` to instantiate,
        then offset the bbox to page coordinates.
        """
        # Fallback generic interactable: find high-contrast boxes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[Interactable] = []
        for c in cnts or []:
            area = cv2.contourArea(c)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(c)
            roi = img[y:y+h, x:x+w]
            inst = cls.detect(roi)
            if inst is None:
                continue
            inst.bbox = (x, y, w, h)
            items.append(inst)
        return items

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional[Interactable]:
        """Instantiate a single generic interactable within the given ROI.
        Does not call detect_all; used by detect_all per-candidate.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return cls(
            id=str(uuid.uuid4()),
            kind="generic",
            bbox=(x, y, w, h),
            meta={}
        )

@dataclass
class Shape(Interactable):
    """
    Generic shape interactable (e.g. rectangle, circle).
    """
    kind: str = field(
        init=False, default="shape"
    )  # subclass of Highlight with distinct intention

    @property
    def shape_type(self) -> str:
        return self.meta.get("shape_type", "unknown")

    def on_click(self) -> Dict[str, Any]:
        self.meta["applied"] = not bool(self.meta.get("applied", False))
        return {"id": self.id, "kind": self.kind, "meta": self.meta}
    
    @classmethod
    def classify_contour(cls, c: np.ndarray) -> Tuple[str, float]:
        area = cv2.contourArea(c)
        # Very high minimum area to only detect major form sections/boxes
        if area < 2000:  # Increased from 500 to drastically reduce noise
            return ("noise", 0.0)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(c)
        rect_area = float(w * h)
        solidity = area / (cv2.contourArea(cv2.convexHull(c)) + 1e-6)
        aspect = w / (h + 1e-6)
        circularity = 4 * np.pi * area / (peri * peri + 1e-6)
        elongated = max(aspect, 1 / (aspect + 1e-6)) > 3.5
        # Filled vs hollow indicator: fraction of bbox covered by contour area
        fill_ratio = area / (rect_area + 1e-6)
        # Approximate stroke thickness heuristic: area to perimeter ratio
        stroke_thickness = area / (peri + 1e-6)
        # Circle
        (cc, _r) = cv2.minEnclosingCircle(c)
        circle_score = float(circularity)  # ~1 for perfect circle
        # Rectangle-like
        rect_score = 0.0
        if len(approx) in (4, 5):
            # Reject elongated (lines) and hollow (low fill) rectangles
            if not elongated and fill_ratio > 0.6 and solidity > 0.9:
                rect_score = (solidity * 0.6) + (min(aspect, 1/aspect) * 0.4)
            else:
                rect_score = 0.0
        # Rounded-rect: many vertices but high solidity and rectangular bounding box fill
        rounded_rect_score = 0.0
        if len(approx) >= 6 and fill_ratio > 0.6 and solidity > 0.9 and not elongated:
            rounded_rect_score = 0.7 * fill_ratio + 0.3 * solidity
        # Polygon generic
        poly_score = 0.0
        if 6 <= len(approx) <= 12 and solidity > 0.8:
            poly_score = 0.5 * solidity + 0.5 * (1 - abs(aspect - 1) * 0.2)
        # Line/arrow-like elongated shapes
        line_score = 0.0
        if elongated and area > 150:
            # Prefer classifying thin elongated shapes as lines, not rectangles
            line_score = 0.6 + 0.4 * min(1.0, stroke_thickness / 10.0)

        # Pick classification with highest score
        scores = [
            (CIRCLE, circle_score),
            (SQUARE, rect_score if 0.85 < aspect < 1.15 else 0.0),
            (RECTANGLE, rect_score),
            (ROUNDED_RECTANGLE, rounded_rect_score),
            (POLYGON, poly_score),
            (LINE, line_score),
        ]
        cls_name, score = max(scores, key=lambda t: t[1])
        return (cls_name, score)

    @classmethod
    def _from_contour(cls, c: np.ndarray, img: np.ndarray) -> Optional["Shape"]:
        area = cv2.contourArea(c)
        # Very high minimum area to only detect major sections
        if area < 2000:  # Increased from 500 to drastically reduce noise
            return None
        # Reject very large shapes (likely section borders or page layout)
        # At 150 DPI, a full page section would be > 50000 pixels
        if area > 50000:
            return None
        (cls_name, score) = cls.classify_contour(c)
        if score <= 0:
            return None
        x, y, w, h = cv2.boundingRect(c)
        # Additional size check: reject shapes wider than 800px or taller than 400px
        # (these are likely page sections, not form elements)
        if w > 800 or h > 400:
            return None
        return cls(
            id=str(uuid.uuid4()),
            bbox=(x, y, w, h),
            meta={"shape_type": cls_name}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional["Shape"]:
        # Robust detection for atypical shapes: circle, rectangle, rounded-rect,
        # polygon, and line/arrow-like elongated shapes.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        edges = cv2.Canny(gray, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts or []:
            item = cls._from_contour(c, img)
            if item is not None:
                return item
        return None

    @classmethod
    def detect_all(cls, img: np.ndarray) -> List[Shape]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        edges = cv2.Canny(gray, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[Shape] = []
        for c in cnts or []:
            item = cls._from_contour(c, img)
            if item is not None:
                items.append(item)
        return items
    
    @classmethod
    def create_bounding_box(
        cls,
        interactable: Interactable,
        shape_type: str,
        **shape_params: Any
    ) -> Shape:
        """
        Factory method to create a Shape interactable over an existing interactable.
        """
        bx, by, bw, bh = interactable.bbox
        iid = str(uuid.uuid4())
        shape_meta = {
            "shape_type": shape_type,
            **shape_params
        }
        return cls(
            id=iid,
            bbox=(bx, by, bw, bh),
            meta=shape_meta
        )

@dataclass
class Text(Interactable):
    kind: str = field(init=False, default="text")

    def on_click(self) -> Dict[str, Any]:
        self.meta["selected"] = not bool(self.meta.get("selected", False))
        self.meta["highlighted"] = not bool(self.meta.get("highlighted", False))

        return {"id": self.id, "kind": self.kind, "meta": self.meta}

    @classmethod
    def _from_contour(cls, c: np.ndarray, img: np.ndarray) -> Optional["Text"]:
        x, y, w, h = cv2.boundingRect(c)
        if w <= 30 or h <= 10:
            return None
        return cls(
            id=str(uuid.uuid4()),
            bbox=(x, y, w, h),
            meta={"selected": False, "highlighted": False}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional["Text"]:
        # High-tolerance text block detection using MSER + morphology
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        mser = cv2.MSER_create()
        mser.setDelta(5)
        mser.setMinArea(100)
        mser.setMaxArea(8000)
        regions, _ = mser.detectRegions(gray)
        mask = np.zeros_like(gray)
        for p in regions:
            hull = cv2.convexHull(p.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], -1, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        merge = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(merge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        for c in cnts:
            item = cls._from_contour(c, img)
            if item is not None:
                return item
        # fallback to largest
        x, y, w, h = cv2.boundingRect(cnts[0])
        return cls(
            id=str(uuid.uuid4()),
            bbox=(x, y, w, h),
            meta={"selected": False, "highlighted": False}
        )

    @classmethod
    def detect_all(cls, img: np.ndarray) -> List[Text]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        # Binary for projection profiles (prefer text dark)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = 255 - bin_img
        # Column separation via vertical projection: split at wide whitespace gutters
        v_proj = inv.sum(axis=0)
        col_gaps = v_proj < (0.05 * inv.shape[0] * 255)
        # Find long gaps (likely between columns)
        cols = []
        start = 0
        for i in range(len(col_gaps)):
            if col_gaps[i]:
                # accumulate gap run
                pass
        # Segment columns by connected runs of non-gaps
        in_run = False
        run_start = 0
        segments = []
        for i, is_gap in enumerate(col_gaps.tolist()):
            if not is_gap and not in_run:
                in_run = True; run_start = i
            elif is_gap and in_run:
                in_run = False; segments.append((run_start, i))
        if in_run:
            segments.append((run_start, len(col_gaps)))
        # Fallback: one segment covering entire width
        if not segments:
            segments = [(0, inv.shape[1])]

        items: List[Text] = []
        # Process each column segment independently to avoid cross-column merges
        for sx, ex in segments:
            col = inv[:, sx:ex]
            # Skip segments that are too narrow
            if ex - sx < 3 or col.shape[0] < 3:
                continue
            # MSER region mask inside the column
            mser = cv2.MSER_create(); mser.setDelta(5); mser.setMinArea(40); mser.setMaxArea(20000)
            col_gray = gray[:, sx:ex]
            # Skip if column is too small for MSER
            if col_gray.shape[0] < 3 or col_gray.shape[1] < 3:
                continue
            regions, _ = mser.detectRegions(col_gray)
            mask = np.zeros_like(col)
            for p in regions:
                hull = cv2.convexHull(p.reshape(-1, 1, 2))
                cv2.drawContours(mask, [hull], -1, 255, -1)
            # Very conservative merging to avoid massive blocks
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))  # Further reduced from (12,2)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # Further reduced from (2,5)
            merge_h = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_kernel, iterations=1)
            merge = cv2.morphologyEx(merge_h, cv2.MORPH_CLOSE, v_kernel, iterations=1)
            cnts, _ = cv2.findContours(merge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Build boxes
            boxes = []
            for c in cnts or []:
                x, y, w, h = cv2.boundingRect(c)
                if w <= 30 or h <= 10:
                    continue
                boxes.append([x + sx, y, x + sx + w, y + h])
            # Merge within column conservatively, but ensure we still build contiguous lines/paragraphs
            def iou(a, b):
                ax0, ay0, ax1, ay1 = a
                bx0, by0, bx1, by1 = b
                ix0, iy0 = max(ax0, bx0), max(ay0, by0)
                ix1, iy1 = min(ax1, bx1), min(ay1, by1)
                iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
                inter = iw * ih
                area_a = (ax1 - ax0) * (ay1 - ay0)
                area_b = (bx1 - bx0) * (by1 - by0)
                union = area_a + area_b - inter + 1e-6
                return inter / union
            merged = True
            while merged and len(boxes) > 1:
                merged = False
                new_boxes = []
                used = [False] * len(boxes)
                for i in range(len(boxes)):
                    if used[i]:
                        continue
                    a = boxes[i]
                    ax0, ay0, ax1, ay1 = a
                    for j in range(i + 1, len(boxes)):
                        if used[j]:
                            continue
                        b = boxes[j]
                        # Same-line horizontal merge
                        vert_overlap = max(0, min(ay1, b[3]) - max(ay0, b[1]))
                        line_height = max(1, min(ay1 - ay0, b[3] - b[1]))
                        same_line = (vert_overlap / line_height) > 0.7
                        close_h = same_line and (abs(b[0] - ax1) < 10 or abs(a[0] - b[2]) < 10)
                        # Paragraph vertical merge requires alignment and small gap
                        left_align = abs(ax0 - b[0]) < 8
                        right_align = abs(ax1 - b[2]) < 8
                        aligned = left_align or right_align
                        vertical_gap = min(abs(b[1] - ay1), abs(a[1] - b[3]))
                        close_v = aligned and (vertical_gap < 4)  # Tightened from 6 to 4
                        # Much stricter size limits to prevent massive blocks
                        merged_w = max(ax1, b[2]) - min(ax0, b[0])
                        merged_h = max(ay1, b[3]) - min(ay0, b[1])
                        merged_area = merged_w * merged_h
                        too_large = merged_area > 50000 or merged_w > 800 or merged_h > 200  # Much stricter limits
                        if not too_large and (iou(a, b) > 0.12 or close_h or close_v):
                            ax0 = min(ax0, b[0]); ay0 = min(ay0, b[1])
                            ax1 = max(ax1, b[2]); ay1 = max(ay1, b[3])
                            used[j] = True
                            merged = True
                    new_boxes.append([ax0, ay0, ax1, ay1])
                    used[i] = True
                boxes = new_boxes
            for x0, y0, x1, y1 in boxes:
                w, h = x1 - x0, y1 - y0
                # Skip massive text blocks - stricter limits
                if w * h > 40000 or w > 700 or h > 150:  # Prevent large page sections
                    continue
                items.append(cls(
                    id=str(uuid.uuid4()),
                    bbox=(x0, y0, w, h),
                    meta={"selected": False, "highlighted": False}
                ))
        return items

    @classmethod
    def create_adjacently(cls, shape: Shape, v_alignment: str, h_alignment: str, offset: float = 1.0) -> Text:
        """
        Factory method to create a Text interactable above a Shape.
        """
        bx, by, bw, bh = shape.bbox
        tid = str(uuid.uuid4())
        text_meta = {
            "selected": False,
            "highlighted": False
        }
        # Expand bbox upwards to cover typical text height
        expanded_bbox = (bx, max(0, by - int(bh * 4)), bw, bh * 5)

        return cls(
            id=tid,
            bbox=expanded_bbox,
            meta=text_meta
        )

@dataclass
class TextField(Interactable):
    """
    Text input field interactable (empty rectangular boxes for form input).
    """
    kind: str = field(init=False, default="textfield")

    def on_click(self) -> Dict[str, Any]:
        self.meta["focused"] = not bool(self.meta.get("focused", False))
        return {"id": self.id, "kind": self.kind, "meta": self.meta}

    @classmethod
    def _from_contour(cls, c: np.ndarray, gray: np.ndarray, edges: np.ndarray, img: np.ndarray) -> Optional["TextField"]:
        area = cv2.contourArea(c)
        # Text fields are typically 100-1000 pixels
        if area < 100 or area > 100000:
            return None

        x, y, w, h = cv2.boundingRect(c)

        # Reject very small boxes (likely noise)
        if w < 30 or h < 10:
            return None

        # Reject very large boxes (likely section borders)
        if w > 500 or h > 80:
            return None

        # Text fields are rectangular (horizontal aspect ratio 1.5-10)
        aspect = w / (h + 1e-6)
        if not (1.5 < aspect < 10):
            return None

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Should be rectangular (4-6 vertices)
        if len(approx) < 4 or len(approx) > 6:
            return None

        bbox_area = w * h
        rect_ratio = area / (bbox_area + 1e-6)

        # Hollow center check - text fields should be empty (low fill ratio)
        if rect_ratio > 0.3:  # Too filled to be an input field
            return None

        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)

        # Thin border (moderate solidity 0.5-0.9)
        if solidity < 0.5 or solidity > 0.9:
            return None

        # Check that edges are concentrated on the perimeter (border), not interior
        pad = max(1, int(min(w, h) * 0.2))
        xi, yi = max(0, x + pad), max(0, y + pad)
        wi, hi = max(1, min(w - 2 * pad, gray.shape[1] - xi)), max(1, min(h - 2 * pad, gray.shape[0] - yi))

        if wi <= 0 or hi <= 0 or xi + wi > edges.shape[1] or yi + hi > edges.shape[0]:
            inner_density = 0.0
        else:
            inner_edges = edges[yi:yi + hi, xi:xi + wi]
            inner_density = float(inner_edges.mean()) if inner_edges.size else 0.0

        border_edges = edges[y:y + h, x:x + w]
        border_density = float(border_edges.mean()) if border_edges.size else 0.0

        # Border should be present but interior should be mostly empty
        if border_density < 5:  # Too faint to be a text field
            return None

        # Interior should have low edge density (empty field)
        if inner_density > border_density * 0.6:  # Too much interior content
            return None

        return cls(
            id=str(uuid.uuid4()),
            bbox=(x, y, w, h),
            meta={"focused": False, "value": ""}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional["TextField"]:
        """Detect a single text field in the given ROI."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)

        # Adaptive threshold to find borders
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        # Edge detection
        edges = cv2.Canny(thr, 40, 140)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts or []:
            item = cls._from_contour(c, gray, edges, img)
            if item is not None:
                return item
        return None

    @classmethod
    def detect_all(cls, img: np.ndarray) -> List["TextField"]:
        """Detect all text fields in the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)

        # Adaptive threshold to find borders
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        # Edge detection
        edges = cv2.Canny(thr, 40, 140)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[TextField] = []
        for c in cnts or []:
            item = cls._from_contour(c, gray, edges, img)
            if item is not None:
                items.append(item)
        return items

@dataclass
class Toggleable(Interactable):
    """
    Toggleable generic.
    """
    kind: str = field(init=False, default="toggleable")

    def on_click(self) -> Dict[str, Any]:
        curr = bool(self.meta.get("checked", False))
        self.meta["checked"] = not curr
        return {"id": self.id, "kind": self.kind, "meta": self.meta}

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional["Toggleable"]:
        raise NotImplementedError

    @classmethod
    def detect_checked_state(cls, **kwargs) -> bool:
        raise False # TODO: not implemented

    @classmethod
    def from_shape(cls, shape: Shape) -> Checkbox | Bubble:
        """
        Factory method to create a Checkbox interactable over an existing Shape.
        """
        bx, by, bw, bh = shape.bbox
        iid = str(uuid.uuid4())
        meta = {
            "checked": cls.detect_checked_state()
        }
        return cls(
            id=iid,
            bbox=(bx, by, bw, bh),
            meta=meta
        )


@dataclass
class Checkbox(Toggleable):
    """
    Toggleable checkbox.
    """
    kind: str = field(init=False, default="checkbox")

    @classmethod
    def _from_contour(cls, c: np.ndarray, gray: np.ndarray, edges: np.ndarray, img: np.ndarray) -> Optional["Checkbox"]:
        area = cv2.contourArea(c)
        # Minimum area for checkboxes
        if area < 25:
            return None
        x, y, w, h = cv2.boundingRect(c)

        # Reject very tiny boxes (likely noise or text fragments)
        if w < 6 or h < 6:
            return None

        # Tighter aspect ratio - checkboxes should be squarish, not elongated like text
        aspect = w / (h + 1e-6)
        if not (0.6 < aspect < 1.7):  # Tightened from 0.5-2.0
            return None

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Checkboxes should have 4-8 vertices (allowing rounded corners)
        if len(approx) < 4 or len(approx) > 8:
            return None

        bbox_area = w * h
        rect_ratio = area / (bbox_area + 1e-6)
        # Moderate rectangle ratio - checkboxes fill their bounding box reasonably well
        if rect_ratio < 0.4:  # Increased from 0.15 to reject text fragments
            return None

        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        # Moderate solidity - checkboxes should be fairly convex
        if solidity < 0.65:  # Increased from 0.5 to reject irregular text shapes
            return None

        # Reasonable size bounds for form checkboxes
        if w > 50 or h > 50:  # Reduced from 80 to avoid large text blocks
            return None

        # Minimum size to avoid tiny noise
        if w < 6 or h < 6:
            return None

        # Verify box-like structure with border density check
        border_edges = edges[y:y + h, x:x + w]
        border_density = float(border_edges.mean()) if border_edges.size else 0.0

        # Require visible borders
        if border_density < 8:  # Increased from 2 to require clearer borders
            return None

        # Detect checked state
        pad2 = max(1, int(min(w, h) * 0.25))
        xi2, yi2 = max(0, x + pad2), max(0, y + pad2)
        wi2, hi2 = max(1, min(w - 2 * pad2, gray.shape[1] - xi2)), max(1, min(h - 2 * pad2, gray.shape[0] - yi2))

        if wi2 <= 0 or hi2 <= 0 or xi2 + wi2 > gray.shape[1] or yi2 + hi2 > gray.shape[0]:
            fill_ratio = 0.0
        else:
            inner = gray[yi2:yi2 + hi2, xi2:xi2 + wi2]
            if inner.size:
                _, inner_bin = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                fill_ratio = (inner_bin == 0).sum() / (inner_bin.size + 1e-6)
            else:
                fill_ratio = 0.0

        checked = fill_ratio > 0.25
        return cls(
            id=str(uuid.uuid4()),
            bbox=(x, y, w, h),
            meta={"checked": checked}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional["Checkbox"]:
        # Detect square checkboxes with balanced preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # Moderate blur to preserve checkbox edges while reducing noise
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        # Adaptive threshold with moderate sensitivity
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        # Moderate edge detection
        edges = cv2.Canny(thr, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts or []:
            item = cls._from_contour(c, gray, edges, img)
            if item is not None:
                return item
        return None

    @classmethod
    def detect_all(cls, img: np.ndarray) -> List[Checkbox]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # Moderate blur to preserve checkbox edges while reducing noise
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        # Adaptive threshold with moderate sensitivity
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        # Moderate edge detection
        edges = cv2.Canny(thr, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[Checkbox] = []
        for c in cnts or []:
            item = cls._from_contour(c, gray, edges, img)
            if item is not None:
                items.append(item)
        return items

@dataclass
class Bubble(Toggleable):
    """
    Toggleable bubble (akin to a radio button).
    """
    kind: str = field(init=False, default="bubble")

    @classmethod
    def _from_contour(cls, c: np.ndarray, gray: np.ndarray, edges: np.ndarray, img: np.ndarray) -> Optional["Bubble"]:
        area = cv2.contourArea(c)
        # Lower minimum area for smaller radio buttons
        if area < 25:
            return None

        # Enforce near-circular shape using circularity and vertex count
        peri = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area) / (peri * peri + 1e-6)
        # Very relaxed circularity threshold to handle distorted/printed circles
        if circularity < 0.5:  # Reduced from 0.65
            return None

        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) <= 5:  # Reduced from 6 - allow slightly more angular shapes
            return None

        (cx, cy), r = cv2.minEnclosingCircle(c)
        x = int(cx); y = int(cy); r = int(max(0, r))

        # Relaxed size bounds: allow tiny to medium bubbles
        if r < 2 or r > 30:  # Increased max from 20 to 30, reduced min from 3 to 2
            return None

        max_r = min(x, y, gray.shape[1] - x - 1, gray.shape[0] - y - 1)
        if max_r <= 0:
            return None
        r = int(min(r, max_r))
        if r <= 0:
            return None

        x0 = max(0, x - r); y0 = max(0, y - r)
        x1 = min(gray.shape[1], x + r); y1 = min(gray.shape[0], y + r)
        if x1 <= x0 or y1 <= y0:
            return None

        roi = edges[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        hh, ww = roi.shape[:2]
        cyi, cxi = hh // 2, ww // 2
        Y, X = np.ogrid[:hh, :ww]
        dist = np.sqrt((Y - cyi) ** 2 + (X - cxi) ** 2)
        ring_mask = (dist >= r * 0.85) & (dist <= r * 1.05)
        inner_mask = dist <= r * 0.55
        ring_values = roi[ring_mask].astype(np.float32) if ring_mask.any() else np.array([], dtype=np.float32)
        inner_values = roi[inner_mask].astype(np.float32) if inner_mask.any() else np.array([], dtype=np.float32)
        ring_density = float(ring_values.mean()) if ring_values.size else 0.0
        inner_density = float(inner_values.mean()) if inner_values.size else 0.0

        # Require ring edges to dominate over overall edges in the ROI to avoid text/boxes
        total_edge = float(roi.astype(np.float32).mean()) if roi.size else 0.0
        ring_edge_ratio = (ring_density / (total_edge + 1e-6)) if total_edge > 0 else 0.0

        pad = max(1, int(r * 0.5))
        xi, yi = int(x - pad), int(y - pad)
        wi, hi = int(pad * 2), int(pad * 2)
        xi = max(0, xi); yi = max(0, yi)
        wi = max(1, min(gray.shape[1] - xi, wi)); hi = max(1, min(gray.shape[0] - yi, hi))
        inner = gray[yi:yi + hi, xi:xi + wi]
        if inner.size == 0:
            return None

        _, inner_bin = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        total = float(inner_bin.size)
        if total <= 0:
            return None
        fill_ratio = (inner_bin == 0).sum() / (total + 1e-6)
        checked = (fill_ratio > 0.25) or (inner_density > 25)

        # Very relaxed constraints for faint radio buttons
        if float(ring_density) < 10:  # Lowered from 20 to catch very faint bubbles
            return None
        if inner_density > ring_density * 0.8:  # Relaxed from 0.7
            return None
        if ring_edge_ratio < 0.25:  # Lowered from 0.35 for very faint bubbles
            return None

        return cls(
            id=str(uuid.uuid4()),
            bbox=(x - r, y - r, r * 2, r * 2),
            meta={"checked": checked}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> Optional["Bubble"]:
        # Robust bubble detection via contours with circular validation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(gray, 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts or []:
            item = cls._from_contour(c, gray, edges, img)
            if item is not None:
                return item
        return None

    @classmethod
    def detect_all(cls, img: np.ndarray) -> List[Bubble]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(gray, 40, 120)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[Bubble] = []
        for c in cnts or []:
            item = cls._from_contour(c, gray, edges, img)
            if item is not None:
                items.append(item)
        return items

INTERACTABLE_TYPES = [Shape, Text, Checkbox, Bubble, TextField]