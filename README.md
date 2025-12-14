# SmartPDF — Intelligent PDF Editor
OCR-based PDF editor intended for working professionals, with features including:
- In-Place Text Editing
- Flattening ⇔ Unflattening (Including Old Documents)
- Interactables (Text, Checkboxes, Bubbles, Images, Signatures, Censors)
- Page Operations (Add, Rotate, Remove)
- Night Filter

Quickstart (Windows PowerShell)
1. Create and activate virtualenv (recommended):
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run the app:
```powershell
python app.py
```
4. Open `http://127.0.0.1:5000` in your browser.

Usage notes
- Upload a PDF using the file input.
- Use Prev/Next to navigate pages.
- Click toolbar actions to place annotations; double-click an annotation to delete it.
- Use the bottom-right handle to resize annotations and drag to move.
- Click "Save to PDF" to write the annotations into a new file `uploads/annotated_<original>.pdf`. A JSON file `annotated_<original>.pdf.json` will be written alongside the PDF to allow re-editing.
 
PDF.js is loaded from CDN by default. If your environment blocks the CDN, place `pdf.js` and `pdf.worker.js` into `static/lib/` (see `static/lib/README.txt`). The app will try CDN first and then fall back to the local copy.

Automatic download (one-line)

If you want the script to fetch the files into `static/lib/` for you, run this from the project root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_pdfjs.ps1
```

After the script finishes, reload the app page in your browser.
Security & production
- This is a demo. For production consider sanitizing filenames, enforcing auth, limiting upload sizes, and serving assets via a proper web server.
