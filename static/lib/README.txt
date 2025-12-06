Download PDF.js build files into this folder if you cannot (or prefer not to) load PDF.js from the CDN.

1. Go to the PDF.js releases or build page and download `pdf.js` and `pdf.worker.js` (or the single bundled `pdf.min.js` and `pdf.worker.min.js`).
   - Example: https://mozilla.github.io/pdf.js/getting_started/

2. Place the files here as:
   - `c:/Users/olive/Projects/SimplyPDF/static/lib/pdf.js`
   - `c:/Users/olive/Projects/SimplyPDF/static/lib/pdf.worker.js`

3. The loader in `templates/index.html` will try the CDN first, then `/static/lib/pdf.js`.

Note: If you save minified files, update `templates/index.html` or `static/main.js` to reference the correct worker path if necessary.
