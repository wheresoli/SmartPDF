from  __future__ import annotations
from typing import Optional, Sequence
import io
import numpy as np
import pypdfium2 as pdfium
from interactables import Interactable
from pypdf import PageObject, PdfWriter, PdfReader
from PIL import Image
import img2pdf
from interactables import INTERACTABLE_TYPES

class Page:
    """Wrapper over pypdf's PageObject with interactable support."""
    def __init__(self, page: PageObject, interactables: Optional[list[Interactable]] = None):
        self.page: PageObject = page
        self.interactables: list[Interactable] = interactables or []
    
    def raycast(self, target: Sequence[int], **kwargs) -> Interactable | None:
        """Return the Interactable at the given (x, y) coordinates.

        Expects `target` as a length-2 (x, y) in pixel coordinates.
        Returns the top-most interactable whose bbox contains the point, or None.
        """
        if target is None:
            return None
        # Accept any array-like and coerce to integers
        try:
            x, y = int(target[0]), int(target[1])
        except Exception:
            return None
        # Later-added interactables considered top-most; iterate in reverse
        for ia in reversed(self.interactables):
            bx, by, bw, bh = ia.bbox
            if (bx <= x < bx + bw) and (by <= y < by + bh):
                return ia
        return None

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
    
    @classmethod
    def unflatten(cls, image: np.ndarray, dpi: int = 150) -> Page:
        """Construct a Page object from a rasterized image array and detect interactables.

        Creates a single-page PDF from the image for consistency, and runs detectors
        to populate `interactables`.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        # Convert to PIL for img2pdf
        if image.ndim == 2:
            pil = Image.fromarray(image, mode="L")
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            arr = image
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError("Unsupported image shape for PDF conversion")

        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        bio.seek(0)
        pdf_bytes = img2pdf.convert([bio.getvalue()])
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_obj = reader.pages[0]

        # Run detectors on the provided image
        detected: list[Interactable] = []
        for interactable_type in INTERACTABLE_TYPES:
            try:
                ia = interactable_type.detect(image)
                detected.append(ia)
            except Exception:
                # Skip detectors that fail; they are reported elsewhere when needed
                continue
        return cls(page=page_obj, interactables=detected)

    @classmethod
    def from_pdf(cls, page: PageObject, dpi: int = 150) -> Page:
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

        detected: list[Interactable] = []
        for interactable_type in INTERACTABLE_TYPES:
            try:
                ia = interactable_type.detect(arr)
                detected.append(ia)
            except Exception:
                continue
        return cls(page=page, interactables=detected)


class Document(Page):
    """Wrapper over pypdf reader/writer with Page wrappers."""
    def __init__(self, pdf: PdfReader | PdfWriter):
        self._pdf: PdfReader | PdfWriter = pdf
        self.pages: list[Page] = []
        self.index: int = 0

    @property
    def page(self) -> Page:
        return self.pages[self.index]

    def wrap_pages(self):
        # For PdfReader: `.pages` is a list of PageObject
        try:
            self.pages = [Page.from_pdf(p) for p in self._pdf.pages]
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

    def raycast(self, target: Sequence[int]) -> Interactable | None:
        """Return the Interactable at the given (x, y) coordinates."""
        super(Document, self).raycast(self.page, target)