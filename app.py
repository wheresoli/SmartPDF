import os
import io
import logging
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_file, render_template, url_for
from werkzeug.utils import secure_filename

from pypdf import PdfReader
import numpy as np

from page import Page, Document


UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROPAGATE_EXCEPTIONS"] = True

# Configure logging to echo errors to terminal
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(name)s: %(message)s')
logger = logging.getLogger("SimplyPDF")
app.logger.setLevel(logging.INFO)


def allowed_file(filename: str) -> bool:
	return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
	return render_template("index.html")


@app.post("/upload")
def upload_pdf():
	if "file" not in request.files:
		return jsonify({"error": "No file part"}), 400
	file = request.files["file"]
	if file.filename == "":
		return jsonify({"error": "No selected file"}), 400
	if not allowed_file(file.filename):
		return jsonify({"error": "Invalid file type"}), 400

	os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
	filename = secure_filename(file.filename)
	path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
	file.save(path)

	try:
		reader = PdfReader(path)
		page_count = len(reader.pages)
	except Exception as e:
		app.logger.exception("Failed to read PDF: %s", e)
		return jsonify({"error": f"Failed to read PDF: {e}"}), 500

	return jsonify({
		"path": filename,
		"pageCount": page_count,
		"pages": [
			{
				"index": i,
				"imageUrl": url_for("page_image", pdf=filename, index=i)
			} for i in range(page_count)
		]
	})


@app.get("/page-image")
def page_image():
	pdf_name = request.args.get("pdf")
	index = request.args.get("index", type=int)
	dpi = request.args.get("dpi", default=150, type=int)
	if not pdf_name or index is None:
		return jsonify({"error": "Missing pdf or index"}), 400
	path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_name)
	if not os.path.exists(path):
		return jsonify({"error": "PDF not found"}), 404

	try:
		reader = PdfReader(path)
		if index < 0 or index >= len(reader.pages):
			return jsonify({"error": "Page index out of range"}), 400
		page_obj = reader.pages[index]
		# Create Page without detection - just for rendering
		page = Page(page_obj)
		img = page.flatten(dpi=dpi)  # np.ndarray RGB
		# Encode to PNG in-memory
		from PIL import Image
		bio = io.BytesIO()
		Image.fromarray(img).save(bio, format="PNG")
		bio.seek(0)
		return send_file(bio, mimetype="image/png")
	except Exception as e:
		app.logger.exception("Rasterization failed: %s", e)
		return jsonify({"error": f"Rasterization failed: {e}"}), 500


@app.get("/detect")
def detect_interactables():
	pdf_name = request.args.get("pdf")
	index = request.args.get("index", type=int)
	dpi = request.args.get("dpi", default=150, type=int)

	# Get detection parameters from query string
	ocr_enabled = request.args.get("ocr", default="1") in ("1", "true", "yes")
	h_kernel = request.args.get("h_kernel", default=10, type=int)
	merge_gap = request.args.get("merge_gap", default=25, type=int)
	min_width = request.args.get("min_width", default=1, type=int)
	min_area = request.args.get("min_area", default=50, type=int)

	if not pdf_name or index is None:
		return jsonify({"error": "Missing pdf or index"}), 400
	path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_name)
	if not os.path.exists(path):
		return jsonify({"error": "PDF not found"}), 404

	app.logger.info(f"Detection params: ocr={ocr_enabled}, h_kernel={h_kernel}, merge_gap={merge_gap}, min_width={min_width}, min_area={min_area}")

	try:
		reader = PdfReader(path)
		if index < 0 or index >= len(reader.pages):
			return jsonify({"error": "Page index out of range"}), 400
		page_obj = reader.pages[index]

		# Use Page.from_pdf to get properly initialized page with Z-ordering
		try:
			page = Page.from_pdf(page_obj, dpi=dpi,
								ocr_enabled=ocr_enabled,
								h_kernel=h_kernel,
								merge_gap=merge_gap,
								min_width=min_width,
								min_area=min_area)
		except Exception as re:
			app.logger.exception("Page.from_pdf failed: %s", re)
			return jsonify({
				"pdf": pdf_name,
				"page": index,
				"interactables": [],
				"error": f"Page processing failed: {re}"
			})

		# Convert all interactables to dict format
		detected: List[Dict[str, Any]] = []
		for ia in page.interactables:
			try:
				ia.page = index
				detected.append(ia.to_dict())
			except Exception as e:
				app.logger.warning("Failed to serialize interactable: %s", e)
				continue

		# Log detection summary to terminal
		if detected:
			counts = {}
			for ia in detected:
				kind = ia.get('kind', 'generic')
				counts[kind] = counts.get(kind, 0) + 1

			app.logger.info(f"Detected {len(detected)} interactables on page {index}: {counts}")

			# Print first few bboxes of each type for debugging
			printed_per_type = {}
			for ia in detected:
				kind = ia.get('kind', 'generic')
				if printed_per_type.get(kind, 0) < 3:  # Print first 3 of each type
					bbox = ia.get('bbox', [])
					z = ia.get('z', 0)
					app.logger.info(f"  {kind} (z={z}): bbox={bbox}")
					printed_per_type[kind] = printed_per_type.get(kind, 0) + 1

		return jsonify({
			"pdf": pdf_name,
			"page": index,
			"interactables": detected
		})
	except Exception as e:
		# Top-level failure (e.g., PDF read issues). Keep response 200 but include error.
		app.logger.exception("Detection failed: %s", e)
		return jsonify({
			"pdf": pdf_name,
			"page": index,
			"interactables": [],
			"error": f"Detection failed: {e}"
		})


@app.post("/client-log")
def client_log():
	"""Accept client-side logs and echo them to server console.

	Expects JSON: { level: "info"|"warn"|"error", message: string, context?: any }
	"""
	try:
		data = request.get_json(silent=True) or {}
		level = (data.get("level") or "info").lower()
		message = str(data.get("message") or "")
		context = data.get("context")
		if level == "error":
			app.logger.error("[CLIENT] %s | context=%s", message, context)
		elif level in ("warn", "warning"):
			app.logger.warning("[CLIENT] %s | context=%s", message, context)
		else:
			app.logger.info("[CLIENT] %s | context=%s", message, context)
		return jsonify({"ok": True})
	except Exception as e:
		app.logger.exception("Client log handling failed: %s", e)
		return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/raycast")
def raycast():
	"""Return interactable under a clicked point (x,y) for a page.

	Query params: pdf, index, x, y, dpi (optional)
	"""
	pdf_name = request.args.get("pdf")
	index = request.args.get("index", type=int)
	x = request.args.get("x", type=int)
	y = request.args.get("y", type=int)
	dpi = request.args.get("dpi", default=150, type=int)
	passthru = request.args.get("passthru", default="true")
	passthru = str(passthru).lower() in ("1", "true", "yes")
	z = request.args.get("z", type=int)
	if not pdf_name or index is None or x is None or y is None:
		return jsonify({"error": "Missing pdf, index, or coordinates"}), 400
	path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_name)
	if not os.path.exists(path):
		return jsonify({"error": "PDF not found"}), 404
	try:
		reader = PdfReader(path)
		if index < 0 or index >= len(reader.pages):
			return jsonify({"error": "Page index out of range"}), 400
		page_obj = reader.pages[index]
		page = Page.from_pdf(page_obj, dpi=dpi)
		# Perform hit-test with passthru/z support
		result = page.raycast((x, y), passthru=passthru, z=z)
		if result is None:
			return jsonify({"pdf": pdf_name, "page": index, "hit": None})
		if isinstance(result, list):
			# Return list of interactables
			out = []
			for ia in result:
				try:
					ia.page = index
					out.append(ia.to_dict())
				except Exception:
					continue
			return jsonify({"pdf": pdf_name, "page": index, "hits": out})
		else:
			result.page = index
			return jsonify({"pdf": pdf_name, "page": index, "hit": result.to_dict()})
	except Exception as e:
		app.logger.exception("Raycast failed: %s", e)
		return jsonify({"error": f"Raycast failed: {e}"}), 500


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)

