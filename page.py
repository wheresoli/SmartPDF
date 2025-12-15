from  __future__ import annotations
from typing import Optional, Sequence, List
import io
import numpy as np
import pypdfium2 as pdfium
from interactables import INTERACTABLE_TYPES, Interactable
from pypdf import PageObject, PdfWriter, PdfReader
from PIL import Image
import img2pdf
from interactables import Shape, Text, Checkbox, Bubble, Z

class Page:
    """Wrapper over pypdf's PageObject with interactable support."""
    def __init__(self, page: PageObject | np.ndarray, interactables: Optional[List[Interactable]] = None):
        self.page: PageObject | np.ndarray = page
        self.interactables: List[Interactable] = interactables or []
    
    @property
    def is_flattened(self) -> bool:
        return isinstance(self.page, np.ndarray)

    def raycast(self, target: Sequence[int], passthru: bool = False, z: int | str | None = None) -> List[Interactable] | Interactable | None:
        """Hit-test interactables at pixel coordinates.

        - `target`: (x, y) point in page pixel coords
        - `passthru`: when True, returns a list of all interactables whose bbox contains the point
        - `z`: optional layer filter; only consider interactables with matching `Interactable.z`

        Returns either the top-most interactable (highest z, latest addition) or a list of all hits.
        """
        if target is None:
            return None
        try:
            x, y = int(target[0]), int(target[1])
        except Exception:
            return None

        hits: List[Interactable] = []
        for ia in self.interactables:
            if ia is None:
                continue
            if z is not None and getattr(ia, 'z', None) != z:
                continue
            bx, by, bw, bh = ia.bbox
            if (bx <= x < bx + bw) and (by <= y < by + bh):
                hits.append(ia)

        if passthru:
            return hits

        if not hits:
            return None
        # Choose the top-most: highest z wins; if equal, prefer later-added
        # Since self.interactables preserves insertion order, iterate reversed for tiebreaker
        best: Interactable | None = None
        best_z = -10**9
        for ia in reversed(self.interactables):
            if ia in hits:
                curr_z = getattr(ia, 'z', 0) if getattr(ia, 'z', None) is not None else 0
                if curr_z >= best_z:
                    best_z = curr_z
                    best = ia
        return best

    @classmethod
    def detect_interactables(cls, img: np.ndarray) -> List[Interactable]:
        detected: List[Interactable] = []
        for interactable_type in INTERACTABLE_TYPES:
            detected.extend(interactable_type.detect(img))
        return detected

    def flatten(self, dpi: int = 150) -> np.ndarray:
        """Rasterize the page into an image array (H x W x 3 uint8).

        Implementation details:
        - Uses `pypdf.PdfWriter` to serialize this page into a single-page PDF.
        - Renders via `pypdfium2` at the requested DPI.
        - Returns a NumPy array in RGB order.

        Requirements: pypdfium2 wheels (no external binaries required).
        """
        writer = PdfWriter()
        writer.add_page(self.page)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        pdf_bytes = buf.getvalue()
        doc = pdfium.PdfDocument(pdf_bytes)
        page = doc.get_page(0)
        # Scale relative to 72 DPI base
        scale = dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        arr = np.array(pil_image)
        # Ensure RGB (drop alpha if present)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        page.close()
        doc.close()
        return arr
    
    def unflatten(self, dpi: int = 150) -> None:
        """Construct a Page object from a rasterized image array and detect interactables.

        Creates a single-page PDF from the image for consistency, and runs detectors
        to populate `interactables`.
        """
        if not isinstance(self.page, np.ndarray):
            raise TypeError("image must be a NumPy array")
        # Convert to PIL for img2pdf
        if self.page.ndim == 2:
            pil = Image.fromarray(self.page, mode="L")
        elif self.page.ndim == 3 and self.page.shape[2] in (3, 4):
            arr = self.page
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError("Unsupported image shape for PDF conversion")

        # Detect interactables directly on the provided image array
        arr = self.page if self.page.ndim != 3 or self.page.shape[2] != 4 else self.page[:, :, :3]
        self.interactables: List[Interactable] = self.detect_interactables(arr)

        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        bio.seek(0)

        pdf_bytes = img2pdf.convert([bio.getvalue()])
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_obj = reader.pages[0]

        self.page = page_obj

    def to_json(self, dpi: int = 150) -> dict:
        """Serialize this Page to a JSON-compatible dictionary.

        Includes the rasterized image and detected interactables.
        """
        if not self.is_flattened:
            img = self.flatten(dpi=dpi)
        else:
            img = self.page

        return {
            "image": img.tolist(),
            "interactables": [ia.to_json() for ia in self.interactables]
        }


    @classmethod
    def from_pdf(cls, page: PageObject, dpi: int = 150, ocr_enabled: bool = True,
                 h_kernel: int = 10, merge_gap: int = 25, min_width: int = 1,
                 min_area: int = 50) -> Page:
        """Construct a Page object from a pypdf PageObject and detect interactables."""
        # Serialize single page to bytes, render via pypdfium2, then run detectors
        writer = PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        doc = pdfium.PdfDocument(buf.getvalue())
        p = doc.get_page(0)
        scale = dpi / 72.0
        bitmap = p.render(scale=scale, rotation=0)
        pil_image = bitmap.to_pil()
        arr = np.array(pil_image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        p.close(); doc.close()

        # NEW APPROACH: Detect all primitives first, then classify based on context
        detected: List[Interactable] = []

        try:
            from primitives import detect_all_rectangles, detect_all_circles, detect_all_text_regions, classify_primitives
            import uuid

            # Step 1: Detect ALL geometric primitives without classification
            all_rects = detect_all_rectangles(arr)
            all_circles = detect_all_circles(arr)
            all_text = detect_all_text_regions(arr, ocr_enabled=ocr_enabled,
                                              h_kernel=h_kernel, merge_gap=merge_gap,
                                              min_width=min_width, min_area=min_area)

            # Step 2: Classify primitives based on features and spatial context
            classified = classify_primitives(all_rects, all_circles, all_text)

            # Step 3: Convert primitives to Interactable objects
            # TEMPORARY: Only detect text for now

            # Text blocks
            for prim in classified['text_blocks']:
                detected.append(Text(
                    id=str(uuid.uuid4()),
                    bbox=prim.bbox,
                    meta={},
                    z=1,
                    primitive=prim
                ))

            # TODO: Re-enable after text detection is working well
            # # Checkboxes
            # for prim in classified['checkboxes']:
            #     x, y, w, h = prim.bbox
            #     # Detect checked state
            #     pad = max(1, int(min(w, h) * 0.25))
            #     if pad < h // 2 and pad < w // 2:
            #         inner = arr[y+pad:y+h-pad, x+pad:x+w-pad]
            #         if inner.size > 0:
            #             gray_inner = inner if inner.ndim == 2 else inner.mean(axis=2).astype(np.uint8)
            #             checked = float(gray_inner.mean()) < 200
            #         else:
            #             checked = False
            #     else:
            #         checked = False

            #     detected.append(Checkbox(
            #         id=str(uuid.uuid4()),
            #         bbox=prim.bbox,
            #         meta={"checked": checked},
            #         z=2
            #     ))

            # # Bubbles
            # for prim in classified['bubbles']:
            #     x, y, w, h = prim.bbox
            #     cx, cy = x + w // 2, y + h // 2
            #     r = (w + h) / 4.0
            #     inner_r = max(1, int(r * 0.5))

            #     # Check if filled
            #     if 0 <= cy-inner_r and cy+inner_r < arr.shape[0] and 0 <= cx-inner_r and cx+inner_r < arr.shape[1]:
            #         inner_region = arr[cy-inner_r:cy+inner_r, cx-inner_r:cx+inner_r]
            #         if inner_region.size > 0:
            #             gray_inner = inner_region if inner_region.ndim == 2 else inner_region.mean(axis=2).astype(np.uint8)
            #             filled = float(gray_inner.mean()) < 200
            #         else:
            #             filled = False
            #     else:
            #         filled = False

            #     detected.append(Bubble(
            #         id=str(uuid.uuid4()),
            #         bbox=prim.bbox,
            #         meta={"selected": filled},
            #         z=2
            #     ))

            # # Shapes (decorative rectangles, borders, etc.)
            # for prim in classified['shapes']:
            #     # Determine shape type from vertices
            #     verts = prim.features.get('vertices', 4)
            #     if verts == 2:
            #         shape_type = 'line'
            #     elif 3 <= verts <= 4:
            #         shape_type = 'rectangle'
            #     else:
            #         shape_type = 'polygon'

            #     detected.append(Shape(
            #         id=str(uuid.uuid4()),
            #         bbox=prim.bbox,
            #         meta={"shape_type": shape_type},
            #         z=0
            #     ))

            # # Text fields (input boxes)
            # try:
            #     from interactables import TextField
            #     for prim in classified['text_fields']:
            #         detected.append(TextField(
            #             id=str(uuid.uuid4()),
            #             bbox=prim.bbox,
            #             meta={},
            #             z=2
            #         ))
            # except Exception:
            #     pass

        except Exception as e:
            # Fallback to old detection if primitive detection fails
            import traceback
            print(f"Primitive detection failed: {e}")
            traceback.print_exc()
            detected = []

        return cls(page=page, interactables=detected)

    # Page does not have a bbox; cropping should be done on Interactable instances.

    def to_document(self, copy: bool = True) -> "Document":
        """Convert this Page into a single-page Document."""
        page = copy.deepcopy(self) if copy else self
        return Document(PdfWriter().add_page(page))

class Document(Page):
    """Wrapper over pypdf reader/writer with Page wrappers."""
    def __init__(self, pdf: PdfReader | PdfWriter):
        self._pdf: PdfReader | PdfWriter = pdf
        self.pages: list[Page] = []
        self.index: int = 0
        # Don't call super().__init__ since Document doesn't have a single page

    @property
    def page(self) -> Page:
        return self.pages[self.index]

    def raycast(self, target: Sequence[int], passthru: bool = False, z: int | None = None) -> Interactable | List[Interactable] | None:
        """Return interactables at the given (x, y) coordinates, with passthru and z filtering."""
        if not self.pages:
            self.wrap_pages()
        return self.page.raycast(target, passthru=passthru, z=z)

    def wrap_pages(self, detect: bool = True):
        """Wrap PDF pages as Page objects.

        Args:
            detect: If True, runs detection on each page. If False, wraps without detection.
        """
        # For PdfReader: `.pages` is a list of PageObject
        try:
            if detect:
                self.pages = [Page.from_pdf(p) for p in self._pdf.pages]
            else:
                # Wrap without detection (for heuristic extraction)
                self.pages = [Page(page=p, interactables=[]) for p in self._pdf.pages]
        except AttributeError:
            self.pages = []

    def delete_page(self, index: int) -> None:
        """Delete a page at the given index (PdfWriter required)."""
        if hasattr(self._pdf, "remove_page"):
            self._pdf.remove_page(index)
            if 0 <= index < len(self.pages):
                del self.pages[index]
        else:
            raise NotImplementedError("delete_page requires a PdfWriter")
    
    def add_page(self, page: Page, index: int | None = None) -> None:
        """Add a page (PdfWriter required). Inserts at index if supported."""
        if hasattr(self._pdf, "insert_page") and index is not None:
            self._pdf.insert_page(index, page.pdf_page)
            self.pages.insert(index, page)
        elif hasattr(self._pdf, "add_page"):
            self._pdf.add_page(page.pdf_page)
            self.pages.append(page)
        else:
            raise NotImplementedError("add_page requires a PdfWriter")

    def flatten(self, dpi: int = 150) -> list[np.ndarray]:
        """Rasterize all pages into image arrays.
        Returns a list of NumPy arrays (H x W x 3 uint8) for each page.
        """
        if not self.pages:
            self.wrap_pages()
        return [p.flatten(dpi=dpi) for p in self.pages]

    @classmethod
    def unflatten(cls, images: list[np.ndarray]) -> "Document":
        """Construct a PDF Document from a list of rasterized page images.

        Uses img2pdf to convert PIL Images into a single PDF, then loads it
        with pypdf.PdfReader and wraps as a Document.
        """
        if not images:
            raise ValueError("unflatten requires at least one image")
        buffers: list[io.BytesIO] = []
        for arr in images:
            if not isinstance(arr, np.ndarray):
                raise TypeError("Each image must be a NumPy array")
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                img = Image.fromarray(arr, mode="RGB")
            else:
                raise ValueError("Unsupported image shape for PDF conversion")
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            bio.seek(0)
            buffers.append(bio)

        pdf_bytes = img2pdf.convert([b.getvalue() for b in buffers])
        reader = PdfReader(io.BytesIO(pdf_bytes))
        doc = cls(reader)
        doc.wrap_pages()
        return doc

    def to_document(self, copy: bool = True) -> "Document":
        return copy.deepcopy(self) if copy else self