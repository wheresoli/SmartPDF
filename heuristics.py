"""
Heuristic extraction - extract ground truth bboxes from color-coded PDFs for algorithm validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import copy
import primitives
import interactables
import page


@dataclass
class HeuristicResult:
    """Result of heuristic extraction containing detected bboxes and metadata."""
    bboxes: List[Tuple[int, int, int, int]]  # List of (x, y, w, h) bounding boxes
    color_used: Tuple[int, int, int]  # RGB color that was detected
    page_index: int  # Which page this is from
    total_count: int  # Number of regions detected


class Heuristic:
    @classmethod
    def generate_from_detection(cls, input_pdf_path: str, output_pdf_path: str,
                               color: Tuple[int, int, int] = (0, 148, 254),
                               dpi: int = 150) -> None:
        """
        Generate a heuristic PDF by running detection and drawing colored boxes.

        This creates a visual representation of what the algorithm detects,
        which can then be manually reviewed and used as ground truth.

        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path where heuristic PDF will be saved
            color: RGB color for the boxes (default: blue)
            dpi: Resolution for rendering

        Example:
            >>> Heuristic.generate_from_detection(
            ...     'uploads/form.pdf',
            ...     'media/heuristic_generated.pdf',
            ...     color=(0, 148, 254)
            ... )
        """
        from pypdf import PdfReader, PdfWriter
        from PIL import Image, ImageDraw
        import io

        # Load and detect
        doc = page.Document(PdfReader(input_pdf_path))
        doc.wrap_pages(detect=True)

        # For each page, render it and draw colored boxes
        writer = PdfWriter()

        for page_idx, pg in enumerate(doc.pages):
            # Flatten page to image
            img_array = pg.flatten(dpi=dpi)
            pil_img = Image.fromarray(img_array)

            # Draw colored rectangles for each detected interactable
            draw = ImageDraw.Draw(pil_img, 'RGBA')

            for ia in pg.interactables:
                x, y, w, h = ia.bbox
                # Draw filled rectangle with transparency
                fill_color = (*color, 100)  # 100/255 alpha
                outline_color = (*color, 255)
                draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=outline_color, width=2)

            # Convert back to PDF page
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # Create PDF page from image
            import img2pdf
            pdf_bytes = img2pdf.convert([img_bytes.getvalue()])
            temp_reader = PdfReader(io.BytesIO(pdf_bytes))
            writer.add_page(temp_reader.pages[0])

            print(f"[Heuristic] Page {page_idx}: Drew {len(pg.interactables)} colored boxes")

        # Save
        with open(output_pdf_path, 'wb') as f:
            writer.write(f)

        print(f"[Heuristic] Generated heuristic PDF: {output_pdf_path}")
        print(f"[Heuristic] Review the boxes and adjust manually if needed")
        print(f"[Heuristic] Then use this as ground truth with extrapolate()")

    @classmethod
    def extrapolate(cls, input: page.Document | page.Page, interactable_type: interactables.Interactable, prim_type: str = 'rect', color: Tuple[int, int, int] | None = None, dpi: int = 150) -> page.Document | page.Page:
        """
        Extrapolate the bboxes of colored rectangles from blocked-out PDFs.
        These bboxes are used for iterating on our detection algorithms.

        Args:
            input: A Document (multi-page) or single Page to extract from
            prim_type: Type of primitive to detect ('rect', 'circle', 'text')
            color: RGB color to detect. If None, auto-detects dominant non-gray color
            dpi: DPI to use for rendering pages

        Returns:
            List of HeuristicResult objects, one per page

        Example:
            >>> from pypdf import PdfReader
            >>> doc = page.Document(PdfReader("media/heuristic_text.pdf"))
            >>> results = Heuristic.extrapolate(doc, prim_type='rect')
            >>> print(f"Found {results[0].total_count} text regions on page 0")
        """
        # Handle both Document and single Page inputs
        if isinstance(input, page.Document):
            pages_to_process = input.pages if input.pages else []
            if not pages_to_process:
                # Wrap pages if not done yet (without detection to avoid slowness)
                input.wrap_pages(detect=False)
                pages_to_process = input.pages

            # Create a deep copy to avoid modifying the original
            result = copy.deepcopy(input)
            result.pages = []

            for page_idx, pg in enumerate(pages_to_process):
                # Create a copy of the page
                pg_copy = page.Page(page=pg.page, interactables=[])

                # Flatten the page to get the rasterized image
                img = pg.flatten(dpi=dpi)

                # Auto-detect color if not provided
                if color is None:
                    detected_color = primitives.find_dominant_color(img, exclude_grayscale=True)
                    print(f"[Heuristic] Auto-detected dominant color on page {page_idx}: RGB{detected_color}")
                else:
                    detected_color = color

                # Detect primitives of the specified type with color filtering
                if prim_type == 'rect':
                    detected_prims = primitives.detect_all_rectangles(img, color=detected_color)
                elif prim_type == 'circle':
                    detected_prims = primitives.detect_all_circles(img, color=detected_color)
                elif prim_type == 'text':
                    detected_prims = primitives.detect_all_text_regions(img, color=detected_color)
                else:
                    raise ValueError(f"Unknown primitive type: {prim_type}")

                # Extract bboxes and create interactables
                bboxes = [prim.bbox for prim in detected_prims]

                import uuid
                for bbox in bboxes:
                    pg_copy.interactables.append(interactable_type(
                        id=str(uuid.uuid4()),
                        bbox=bbox,
                        meta={},
                        z=1
                    ))

                print(f"[Heuristic] Page {page_idx}: Detected {len(bboxes)} {prim_type} regions with color RGB{detected_color}")

                result.pages.append(pg_copy)
        else:
            # Single page input
            result = page.Page(page=input.page, interactables=[])

            # Flatten the page to get the rasterized image
            img = input.flatten(dpi=dpi)

            # Auto-detect color if not provided
            if color is None:
                detected_color = primitives.find_dominant_color(img, exclude_grayscale=True)
                print(f"[Heuristic] Auto-detected dominant color: RGB{detected_color}")
            else:
                detected_color = color

            # Detect primitives of the specified type with color filtering
            if prim_type == 'rect':
                detected_prims = primitives.detect_all_rectangles(img, color=detected_color)
            elif prim_type == 'circle':
                detected_prims = primitives.detect_all_circles(img, color=detected_color)
            elif prim_type == 'text':
                detected_prims = primitives.detect_all_text_regions(img, color=detected_color)
            else:
                raise ValueError(f"Unknown primitive type: {prim_type}")

            # Extract bboxes and create interactables
            bboxes = [prim.bbox for prim in detected_prims]

            import uuid
            for bbox in bboxes:
                result.interactables.append(interactable_type(
                    id=str(uuid.uuid4()),
                    bbox=bbox,
                    meta={},
                    z=1
                ))

            print(f"[Heuristic] Detected {len(bboxes)} {prim_type} regions with color RGB{detected_color}")

        return result

    @classmethod
    def compare_to_ground_truth(cls, input: page.Document | page.Page,
                                ground_truth: page.Document | page.Page,
                                iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Compare detected bboxes against ground truth heuristic bboxes.

        Args:
            detected_bboxes: List of detected (x, y, w, h) bboxes
            ground_truth: HeuristicResult containing ground truth bboxes
            iou_threshold: IoU threshold for considering a match

        Returns:
            Dictionary with metrics: precision, recall, f1, avg_iou
        """
        def bbox_iou(box1, box2):
            """Calculate IoU between two bboxes."""
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            intersection = (x_right - x_left) * (y_bottom - y_top)
            union = w1 * h1 + w2 * h2 - intersection

            return intersection / (union + 1e-6)

        # Handle both Document and Page inputs
        if isinstance(input, page.Document) and isinstance(ground_truth, page.Document):
            # Multi-page comparison - compare corresponding pages
            all_metrics = []
            for page_idx, (a_page, b_page) in enumerate(zip(input.pages, ground_truth.pages)):
                a_bboxes = [ia.bbox for ia in a_page.interactables]
                b_bboxes = [ia.bbox for ia in b_page.interactables]

                # Calculate IoU for each pair of bboxes
                if len(a_bboxes) == 0 or len(b_bboxes) == 0:
                    # No bboxes to compare
                    page_metrics = {
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'avg_iou': 0.0,
                        'true_positives': 0,
                        'false_positives': len(a_bboxes),
                        'false_negatives': len(b_bboxes)
                    }
                else:
                    iou_matrix = np.zeros((len(a_bboxes), len(b_bboxes)))
                    for i, a_bbox in enumerate(a_bboxes):
                        for j, b_bbox in enumerate(b_bboxes):
                            iou_matrix[i, j] = bbox_iou(a_bbox, b_bbox)

                    # Match bboxes based on IoU threshold (greedy matching)
                    matched_a = set()
                    matched_b = set()
                    ious = []

                    # Sort by IoU descending
                    iou_pairs = []
                    for i in range(len(a_bboxes)):
                        for j in range(len(b_bboxes)):
                            if iou_matrix[i, j] >= iou_threshold:
                                iou_pairs.append((iou_matrix[i, j], i, j))

                    iou_pairs.sort(reverse=True)

                    for iou_val, i, j in iou_pairs:
                        if i not in matched_a and j not in matched_b:
                            matched_a.add(i)
                            matched_b.add(j)
                            ious.append(iou_val)

                    true_positives = len(matched_a)
                    false_positives = len(a_bboxes) - true_positives
                    false_negatives = len(b_bboxes) - len(matched_b)

                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    avg_iou = sum(ious) / len(ious) if ious else 0.0

                    page_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'avg_iou': avg_iou,
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'false_negatives': false_negatives
                    }

                all_metrics.append(page_metrics)

            # Average metrics across all pages
            if all_metrics:
                avg_metrics = {
                    'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
                    'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
                    'f1': sum(m['f1'] for m in all_metrics) / len(all_metrics),
                    'avg_iou': sum(m['avg_iou'] for m in all_metrics) / len(all_metrics),
                    'true_positives': sum(m['true_positives'] for m in all_metrics),
                    'false_positives': sum(m['false_positives'] for m in all_metrics),
                    'false_negatives': sum(m['false_negatives'] for m in all_metrics)
                }
                return avg_metrics
            else:
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'avg_iou': 0.0,
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0
                }
        else:
            # Single page comparison
            a_page = input if isinstance(input, page.Page) else input.pages[0]
            b_page = ground_truth if isinstance(ground_truth, page.Page) else ground_truth.pages[0]

            a_bboxes = [ia.bbox for ia in a_page.interactables]
            b_bboxes = [ia.bbox for ia in b_page.interactables]

            if len(a_bboxes) == 0 or len(b_bboxes) == 0:
                true_positives = 0
                false_positives = len(a_bboxes)
                false_negatives = len(b_bboxes)
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                avg_iou = 0.0
            else:
                # Calculate IoU for each pair of bboxes
                iou_matrix = np.zeros((len(a_bboxes), len(b_bboxes)))
                for i, a_bbox in enumerate(a_bboxes):
                    for j, b_bbox in enumerate(b_bboxes):
                        iou_matrix[i, j] = bbox_iou(a_bbox, b_bbox)

                # Match bboxes based on IoU threshold (greedy matching)
                matched_a = set()
                matched_b = set()
                ious = []

                # Sort by IoU descending
                iou_pairs = []
                for i in range(len(a_bboxes)):
                    for j in range(len(b_bboxes)):
                        if iou_matrix[i, j] >= iou_threshold:
                            iou_pairs.append((iou_matrix[i, j], i, j))

                iou_pairs.sort(reverse=True)

                for iou_val, i, j in iou_pairs:
                    if i not in matched_a and j not in matched_b:
                        matched_a.add(i)
                        matched_b.add(j)
                        ious.append(iou_val)

                true_positives = len(matched_a)
                false_positives = len(a_bboxes) - true_positives
                false_negatives = len(b_bboxes) - len(matched_b)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                avg_iou = sum(ious) / len(ious) if ious else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou': avg_iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }