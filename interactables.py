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
        if area < 200:
            return ("noise", 0.0)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(c)
        rect_area = float(w * h)
        solidity = area / (cv2.contourArea(cv2.convexHull(c)) + 1e-6)
        aspect = w / (h + 1e-6)
        circularity = 4 * np.pi * area / (peri * peri + 1e-6)
        elongated = max(aspect, 1 / (aspect + 1e-6)) > 3.5
        # Circle
        (cc, _r) = cv2.minEnclosingCircle(c)
        circle_score = float(circularity)  # ~1 for perfect circle
        # Rectangle-like
        rect_score = 0.0
        if len(approx) in (4, 5):
            rect_score = (solidity * 0.6) + (min(aspect, 1/aspect) * 0.4)
        # Rounded-rect: many vertices but high solidity and rectangular bounding box fill
        fill_ratio = area / (rect_area + 1e-6)
        rounded_rect_score = 0.0
        if len(approx) >= 6 and fill_ratio > 0.6 and solidity > 0.9:
            rounded_rect_score = 0.7 * fill_ratio + 0.3 * solidity
        # Polygon generic
        poly_score = 0.0
        if 6 <= len(approx) <= 12 and solidity > 0.8:
            poly_score = 0.5 * solidity + 0.5 * (1 - abs(aspect - 1) * 0.2)
        # Line/arrow-like elongated shapes
        line_score = 0.0
        if elongated and area > 150:
            line_score = 0.6 * elongated + 0.4 * (solidity)

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
        if area < 200:
            return None
        (cls_name, score) = cls.classify_contour(c)
        if score <= 0:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return cls(
            id=str(uuid.uuid4()),
            bbox=(x, y, w, h),
            meta={"shape_type": cls_name}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> list[Shape]:
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
    def detect(cls, img: np.ndarray) -> list[Text]:
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
        items: List[Text] = []
        for c in cnts or []:
            item = cls._from_contour(c, img)
            if item is not None:
                items.append(item)
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
    def detect(cls, img: np.ndarray) -> list[Toggleable]:
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
        if area < 40:
            return None
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        if not (0.85 < aspect < 1.15):
            return None
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4:
            return None
        bbox_area = w * h
        rect_ratio = area / (bbox_area + 1e-6)
        if rect_ratio < 0.5:
            return None
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        if solidity < 0.8:
            return None
        pad = max(1, int(min(w, h) * 0.12))
        xi, yi = x + pad, y + pad
        wi, hi = max(1, w - 2 * pad), max(1, h - 2 * pad)
        inner_edges = edges[yi:yi + hi, xi:xi + wi]
        border_edges = edges[y:y + h, x:x + w]
        inner_density = float(inner_edges.mean()) if inner_edges.size else 0.0
        border_density = float(border_edges.mean()) if border_edges.size else 0.0
        if border_density < inner_density * 1.5:
            return None
        pad2 = max(1, int(min(w, h) * 0.22))
        xi2, yi2 = x + pad2, y + pad2
        wi2, hi2 = max(1, w - 2 * pad2), max(1, h - 2 * pad2)
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
    def detect(cls, img: np.ndarray) -> list[Checkbox]:
        # Detect square checkboxes with tolerant preprocessing but stricter validation
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        # Invert to make dark borders foreground under adaptive threshold
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        # Strengthen edges and close small gaps
        edges = cv2.Canny(thr, 40, 140)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
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
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        edges = cv2.Canny(thr, 40, 140)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
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
        if area < 40:
            return None
        # Enforce near-circular shape using circularity and vertex count
        peri = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area) / (peri * peri + 1e-6)
        if circularity < 0.75:
            return None
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) <= 6:  # too few vertices -> likely polygon/square
            return None
        (cx, cy), r = cv2.minEnclosingCircle(c)
        x = int(cx); y = int(cy); r = int(max(0, r))
        if r <= 0:
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
        # Strong constraints to avoid non-bubble contours
        if float(ring_density) < 35:
            return None
        if inner_density > ring_density * 0.7:
            return None
        if ring_edge_ratio < 0.5:
            return None
        return cls(
            id=str(uuid.uuid4()),
            bbox=(x - r, y - r, r * 2, r * 2),
            meta={"checked": checked}
        )

    @classmethod
    def detect(cls, img: np.ndarray) -> list[Bubble]:
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

INTERACTABLE_TYPES = [Shape, Text, Checkbox, Bubble]