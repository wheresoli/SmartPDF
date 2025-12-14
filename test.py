"""Unified test suite for text detection."""
import sys
import cv2
import numpy as np
from pypdf import PdfReader
from page import Document


def diagnose_easyocr():
    """Diagnostic test for EasyOCR installation and functionality."""
    print("="*70)
    print("EASYOCR DIAGNOSTIC")
    print("="*70)

    print(f"\nPython executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        import easyocr
        print("\n[PASS] EasyOCR imported successfully")
        print(f"       Version: {easyocr.__version__}")
    except ImportError as e:
        print(f"\n[FAIL] EasyOCR import failed: {e}")
        print("\nTo install: pip install easyocr")
        return False

    try:
        print("\n[TEST] Initializing EasyOCR Reader...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[PASS] Reader initialized successfully")
    except Exception as e:
        print(f"[FAIL] Reader initialization failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Create a simple test image
        img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        print("\n[TEST] Running test detection...")
        results = reader.readtext(img)
        print(f"[PASS] Detection completed: {len(results)} results")
        if results:
            for _, text, conf in results:
                print(f"       Found: '{text}' (confidence: {conf:.2f})")
    except Exception as e:
        print(f"[FAIL] Detection failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[PASS] All EasyOCR tests passed!")
    print("       If Flask server shows 'CV fallback', restart it: python app.py")
    print("\n" + "="*70)
    return True


def extract_ground_truth_mask(img_path='media/heuristic_text.png'):
    """Extract ground truth mask from manually-drawn blue boxes.

    Returns:
        Binary mask where blue pixels = 255, everything else = 0
    """
    img = cv2.imread(img_path)
    # Extract blue channel mask (blue boxes you manually drew)
    # Blue is RGB(0, 148, 254) - isolate blue pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Blue hue range in HSV (roughly 100-130 degrees)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def detect_text_regions(pdf_path='media/heuristic_none.pdf', debug=False):
    """Run SimplyPDF's actual text detection on flattened PDF.

    Returns:
        List of (x, y, w, h) tuples
    """
    doc = Document(PdfReader(pdf_path))
    doc.wrap_pages(detect=True)

    if debug:
        print(f"\n  Debug: {len(doc.pages[0].interactables)} total detections")
        print("  First 10 detections:")
        for i, ia in enumerate(doc.pages[0].interactables[:10]):
            x, y, w, h = ia.bbox
            text = ""
            if hasattr(ia, 'primitive') and hasattr(ia.primitive, 'features'):
                text = ia.primitive.features.get('text', '')
            print(f"    {i+1:2d}. [{w:4d}x{h:2d} @ ({x:4d},{y:4d})] {text[:50]}")

    return [ia.bbox for ia in doc.pages[0].interactables]


def create_detection_mask(bboxes, img_shape):
    """Create binary mask from detected bboxes.

    Args:
        bboxes: List of (x, y, w, h) tuples
        img_shape: (height, width)

    Returns:
        Binary mask where detected regions = 255
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    for x, y, w, h in bboxes:
        mask[y:y+h, x:x+w] = 255
    return mask


def compare_masks(gt_mask, det_mask):
    """Compare two binary masks pixel-by-pixel.

    Returns:
        Dict with 'iou', 'precision', 'recall', 'f1' keys
    """
    intersection = np.logical_and(gt_mask, det_mask).sum()
    union = np.logical_or(gt_mask, det_mask).sum()

    gt_pixels = gt_mask.sum() / 255  # Count of white pixels
    det_pixels = det_mask.sum() / 255

    iou = intersection / (union + 1e-6)
    precision = intersection / (det_pixels + 1e-6) if det_pixels > 0 else 0
    recall = intersection / (gt_pixels + 1e-6) if gt_pixels > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'gt_pixels': int(gt_pixels),
        'det_pixels': int(det_pixels),
        'overlap_pixels': int(intersection)
    }


def run_full_test():
    """Run complete test suite."""
    print("="*70)
    print("TEXT DETECTION TEST SUITE")
    print("="*70)

    # 1. Extract GT mask from blue pixels in heuristic_text.png
    print("\n[1/3] Extracting ground truth from heuristic_text.png...")
    gt_mask = extract_ground_truth_mask()
    cv2.imwrite('gt_mask.png', gt_mask)
    gt_pixel_count = gt_mask.sum() / 255
    print(f"      GT mask saved: {int(gt_pixel_count)} blue pixels")

    # 2. Run detection on heuristic_none.pdf
    print("\n[2/3] Running text detection on heuristic_none.pdf...")
    det_boxes = detect_text_regions(debug=True)
    print(f"      Detected: {len(det_boxes)} text regions")

    # Show ALL detected boxes
    print(f"\n  ALL {len(det_boxes)} detected boxes:")
    for i, (x, y, w, h) in enumerate(det_boxes):
        print(f"    {i+1:2d}. x={x:4d} y={y:4d} w={w:4d} h={h:3d}")

    # 3. Create detection mask
    print("\n[3/3] Creating detection mask...")
    img_shape = gt_mask.shape
    det_mask = create_detection_mask(det_boxes, img_shape)
    cv2.imwrite('det_mask.png', det_mask)
    det_pixel_count = det_mask.sum() / 255
    print(f"      Detection mask saved: {int(det_pixel_count)} pixels covered")

    # 4. Compare masks pixel-by-pixel
    print("\n" + "="*70)
    print("PIXEL-LEVEL MASK COMPARISON")
    print("="*70)
    metrics = compare_masks(gt_mask, det_mask)

    print(f"\nGround Truth:  {metrics['gt_pixels']:,} pixels")
    print(f"Detected:      {metrics['det_pixels']:,} pixels")
    print(f"Overlap:       {metrics['overlap_pixels']:,} pixels")

    print(f"\nIoU:           {metrics['iou']*100:.2f}%")
    print(f"Precision:     {metrics['precision']*100:.2f}%")
    print(f"Recall:        {metrics['recall']*100:.2f}%")
    print(f"F1 Score:      {metrics['f1']*100:.2f}%")

    # 5. Create visual comparison
    print("\n" + "="*70)
    print("CREATING VISUAL COMPARISON")
    print("="*70)

    # Load original image
    img = cv2.imread('media/heuristic_none.png')

    # Create RGB comparison: red=GT only, green=detected only, yellow=overlap
    comparison = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    comparison[:, :, 2] = np.where((gt_mask > 0) & (det_mask == 0), 255, 0)  # Red: GT only
    comparison[:, :, 1] = np.where((det_mask > 0) & (gt_mask == 0), 255, 0)  # Green: detected only
    comparison[:, :, 0] = np.where((gt_mask > 0) & (det_mask > 0), 255, 0)   # Blue: overlap
    comparison[:, :, 1] = np.where((gt_mask > 0) & (det_mask > 0), 255, comparison[:, :, 1])  # Yellow = red+green

    # Blend with original image
    overlay = cv2.addWeighted(img, 0.6, comparison, 0.4, 0)
    cv2.imwrite('comparison_overlay.png', overlay)
    print("      Saved comparison_overlay.png (red=GT only, green=det only, yellow=overlap)")

    # Also create a simple bbox visualization to verify det_mask is correct
    bbox_viz = img.copy()
    for x, y, w, h in det_boxes:
        cv2.rectangle(bbox_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('bbox_visualization.png', bbox_viz)
    print("      Saved bbox_visualization.png (green boxes = detected)")

    print("\n" + "="*70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'diagnose':
        # Run EasyOCR diagnostic
        diagnose_easyocr()
    else:
        # Run full test suite
        run_full_test()
