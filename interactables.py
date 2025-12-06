from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid
import cv2


@dataclass
class Interactable:
    id: str
    kind: str  # e.g. "highlight", "censor", "checkbox", "bubble"
    page: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixel coords
    meta: Dict[str, Any]

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
        d["page"] = int(d.get("page", 0))
        return d

    def on_click(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "meta": self.meta}

    @classmethod
    def detect(cls, img: np.ndarray) -> Interactable:
        """Attempt to identify this pattern of interactable from the given image."""
        # Fallback generic interactable: find largest high-contrast box
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise ValueError("No interactable-like regions found")
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return cls(
            id=str(uuid.uuid4()),
            kind="generic",
            page=0,
            bbox=(x, y, w, h),
            meta={}
        )

@dataclass
class Shape(Interactable):
    """
    Generic shape interactable (e.g. rectangle, circle).
    """
    kind: str = field(
        init=False, default="selected"
    )  # subclass of Highlight with distinct intention

    def on_click(self) -> Dict[str, Any]:
        self.meta["applied"] = not bool(self.meta.get("applied", False))
        return {"id": self.id, "kind": self.kind, "meta": self.meta}

    @classmethod
    def detect(cls, img: np.ndarray) -> Shape:
        # Robust detection for atypical shapes: circle, rectangle, rounded-rect,
        # polygon, and line/arrow-like elongated shapes.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.bilateralFilter(gray, 7, 75, 75)
        edges = cv2.Canny(gray, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise ValueError("No shape-like region detected")

        def classify_contour(c: np.ndarray) -> Tuple[str, float]:
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
                ("circle", circle_score),
                ("rect", rect_score),
                ("rounded-rect", rounded_rect_score),
                ("polygon", poly_score),
                ("line", line_score),
            ]
            cls_name, score = max(scores, key=lambda t: t[1])
            return (cls_name, score)

        best_box = None
        best_meta = None
        best_score = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 200:
                continue
            (cls_name, score) = classify_contour(c)
            if score <= 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)
                best_meta = {"shape_type": cls_name}

        if best_box is None:
            raise ValueError("No atypical shapes classified")
        x, y, w, h = best_box
        return cls(
            id=str(uuid.uuid4()),
            page=0,
            bbox=(x, y, w, h),
            meta=best_meta
        )
    
    @classmethod
    def create_shape_over_interactable(
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
            page=interactable.page,
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
    def detect(cls, img: np.ndarray) -> Text:
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
            raise ValueError("No text regions detected")
        # Prefer wide blocks resembling lines/paragraphs
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 30 and h > 10:
                return cls(
                    id=str(uuid.uuid4()),
                    page=0,
                    bbox=(x, y, w, h),
                    meta={"selected": False, "highlighted": False}
                )
        # fallback to largest
        x, y, w, h = cv2.boundingRect(cnts[0])
        return cls(
            id=str(uuid.uuid4()),
            page=0,
            bbox=(x, y, w, h),
            meta={"selected": False, "highlighted": False}
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
    def detect(cls, img: np.ndarray) -> Checkbox:
        raise NotImplementedError


@dataclass
class Checkbox(Interactable):
    """
    Toggleable checkbox.
    """
    kind: str = field(init=False, default="checkbox")

    @classmethod
    def detect(cls, img: np.ndarray) -> Checkbox:
        # Detect square checkboxes with more tolerant preprocessing
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
        best = None
        best_score = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 40:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / (h + 1e-6)
            if 0.7 < aspect < 1.3:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) in (4, 5):
                    # Prefer sharper borders and small size typical of checkboxes
                    border_roi = gray[max(0, y-2):min(gray.shape[0], y+h+2), max(0, x-2):min(gray.shape[1], x+w+2)]
                    var = float(np.var(border_roi)) if border_roi.size else 0.0
                    size_penalty = min(w, h)
                    score = var / (1 + abs(1 - aspect)) - 0.001 * size_penalty
                    if score > best_score:
                        best_score = score
                        best = (x, y, w, h)
        if best is None:
            raise ValueError("No checkbox detected")
        x, y, w, h = best
        # Estimate checked state using inner region darkness
        pad = max(1, int(min(w, h) * 0.22))
        xi, yi = x + pad, y + pad
        wi, hi = max(1, w - 2 * pad), max(1, h - 2 * pad)
        inner = gray[yi:yi + hi, xi:xi + wi]
        if inner.size:
            _, inner_bin = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fill_ratio = (inner_bin == 0).sum() / (inner_bin.size + 1e-6)
        else:
            fill_ratio = 0.0
        checked = fill_ratio > 0.25
        return cls(
            id=str(uuid.uuid4()),
            page=0,
            bbox=(x, y, w, h),
            meta={"checked": checked}
        )


@dataclass
class Bubble(Toggleable):
    """
    Toggleable bubble (akin to a radio button).
    """
    kind: str = field(init=False, default="bubble")

    @classmethod
    def detect(cls, img: np.ndarray) -> Bubble:
        # Robust bubble (radio) detection: circles with border/inner dot
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(gray, 40, 120)
        # Hough circle detection tolerant to compression
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                                   param1=120, param2=25, minRadius=6, maxRadius=100)
        if circles is None:
            raise ValueError("No bubble-like circle detected")
        circles = np.around(circles).astype(int)[0]
        # Pick the most prominent circle by edge density
        best = None
        best_score = -1.0
        for x, y, r in circles:
            x = int(x); y = int(y); r = int(max(0, r))
            if r <= 0:
                continue
            # Clamp radius so circle fits entirely within image bounds
            max_r = min(x, y, gray.shape[1] - x - 1, gray.shape[0] - y - 1)
            if max_r <= 0:
                continue
            r = int(min(r, max_r))
            if r <= 0:
                continue
            x0 = max(0, x - r); y0 = max(0, y - r)
            x1 = min(gray.shape[1], x + r); y1 = min(gray.shape[0], y + r)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = edges[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            # Use float dtype to avoid integer overflow in reductions
            score = float(roi.astype(np.float32).mean())
            if score > best_score:
                best_score = score
                best = (x, y, r)
        if best is None:
            raise ValueError("No suitable bubble circle ROI")
        bx, by, r = best
        # Determine selected by inner dark region ratio
        pad = max(1, int(r * 0.5))
        xi, yi = int(bx - pad), int(by - pad)
        wi, hi = int(pad * 2), int(pad * 2)
        xi = max(0, xi); yi = max(0, yi)
        wi = max(1, min(gray.shape[1] - xi, wi)); hi = max(1, min(gray.shape[0] - yi, hi))
        inner = gray[yi:yi + hi, xi:xi + wi]
        if inner.size == 0:
            raise ValueError("Inner bubble region is empty")
        # Otsu threshold; if variance is low, guard against degenerate results
        _, inner_bin = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        total = float(inner_bin.size)
        if total <= 0:
            raise ValueError("Inner bubble region size invalid")
        fill_ratio = (inner_bin == 0).sum() / (total + 1e-6)
        checked = fill_ratio > 0.20
        return cls(
            id=str(uuid.uuid4()),
            page=0,
            bbox=(bx - r, by - r, r * 2, r * 2),
            meta={"checked": checked}
        )

INTERACTABLE_TYPES = [Text, Shape, Checkbox, Bubble]