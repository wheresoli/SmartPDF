(() => {
    const state = {
        pdf: null,
        pageCount: 0,
        pageIndex: 0,
        pages: [],
        interactables: [],
        dpi: 200,
        overlaysVisible: false,
    };

    const el = {
        fileInput: document.getElementById('fileInput'),
        pagePrev: document.getElementById('btnPrev'),
        pageNext: document.getElementById('btnNext'),
        pageNum: document.getElementById('pageNum'),
        pageCount: document.getElementById('pageCount'),
        detectToggle: document.getElementById('btnDetectionsToggle'),
        detectRescan: document.getElementById('btnDetectionsRescan'),
        scaleInput: document.getElementById('scale') || document.getElementById('scaleInput') || document.getElementById('scaleSlider'),
        canvasContainer: document.getElementById('canvasContainer'),
        canvas: null,
        ctx: null,
        debugList: null
    };

    function ensureCanvas() {
        if (!el.canvas) {
            el.canvas = document.createElement('canvas');
            el.canvas.id = 'pageCanvas';
            el.canvas.style.maxWidth = '100%';
            el.canvas.style.boxShadow = '0 1px 3px rgba(0,0,0,0.2)';
            el.canvasContainer.innerHTML = '';
            el.canvasContainer.appendChild(el.canvas);
            el.ctx = el.canvas.getContext('2d');
        }
    }

    function renderPageInfo() {
        el.pageNum.value = (state.pageIndex + 1);
        el.pageCount.textContent = `/ ${state.pageCount || 0}`;
    }

    function clearCanvas() {
        el.ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
    }

    async function loadImageForPage() {
        if (!state.pdf) return;
        const page = state.pages[state.pageIndex];
        if (!page) return;
        const img = new Image();
        img.crossOrigin = 'anonymous';
        // Request the server-side raster at current DPI
        const url = `/page-image?pdf=${encodeURIComponent(state.pdf)}&index=${state.pageIndex}&dpi=${state.dpi}`;
        img.src = url;
        await new Promise((res, rej) => {
            img.onload = res;
            img.onerror = rej;
        });
        ensureCanvas();
        el.canvas.width = img.naturalWidth;
        el.canvas.height = img.naturalHeight;
        clearCanvas();
        el.ctx.drawImage(img, 0, 0);
        // If overlays are toggled on, redraw them on top
        if (state.overlaysVisible && state.interactables && state.interactables.length) {
            drawInteractables();
        }
    }

    function drawInteractables() {
        if (!state.interactables || !state.interactables.length) return;
        const colors = {
            Text: '#1f77b4',
            Shape: '#ff7f0e',
            Checkbox: '#2ca02c',
            Bubble: '#d62728',
            generic: '#9467bd'
        };
        state.interactables.forEach(ia => {
            const kind = ia.kind || 'generic';
            const color = colors[kind] || '#000';
            const bbox = ia.bbox;
            if (!bbox) return;
            const [x, y, w, h] = bbox;
            el.ctx.save();
            el.ctx.strokeStyle = color;
            el.ctx.lineWidth = 2;
            el.ctx.setLineDash([6, 4]);
            el.ctx.strokeRect(x, y, w, h);
            el.ctx.fillStyle = color;
            el.ctx.globalAlpha = 0.12;
            el.ctx.fillRect(x, y, w, h);
            el.ctx.restore();
        });
    }

    function renderDebugList() {
        // Render as an overlay tooltip list if desired; for now, console log for debugging
        const count = state.interactables?.length || 0;
        console.group(`Interactables (${count})`);
        (state.interactables || []).forEach((ia, i) => {
            if (ia?.meta?.error) {
                console.warn(`${i + 1}. ${ia.kind} error:`, ia.meta.error);
            } else {
                console.log(`${i + 1}.`, ia);
            }
        });
        console.groupEnd();
    }

    async function uploadPdf(file) {
        const fd = new FormData();
        fd.append('file', file);
        const resp = await fetch('/upload', {
            method: 'POST',
            body: fd
        });
        if (!resp.ok) throw new Error('Upload failed');
        const data = await resp.json();
        state.pdf = data.path;
        state.pageCount = data.pageCount;
        state.pages = data.pages;
        state.pageIndex = 0;
        renderPageInfo();
        // No title element; keep toolbar simple
        await loadImageForPage();
        // Auto-rescan once after upload to populate detections
        try {
            await detectForPage();
            state.overlaysVisible = true;
            await loadImageForPage();
            drawInteractables();
        } catch (e) {
            // Keep going if detection fails; user can rescan manually
            state.interactables = [];
            state.overlaysVisible = false;
            console.warn('Initial detect failed:', e);
        }
        renderDebugList();
    }

    async function detectForPage() {
        if (!state.pdf) return;
        const url = `/detect?pdf=${encodeURIComponent(state.pdf)}&index=${state.pageIndex}&dpi=${state.dpi}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('Detect failed');
        const data = await resp.json();
        if (data && data.error) {
            console.warn('Detection error:', data.error);
        }
        state.interactables = data.interactables || [];
        // Keep current image; just draw overlays if toggled on
        if (state.overlaysVisible) {
            // Redraw base image then overlays to ensure clean stacking
            await loadImageForPage();
            drawInteractables();
        }
        renderDebugList();
    }

    async function goto(delta) {
        if (!state.pdf) return;
        const next = state.pageIndex + delta;
        if (next < 0 || next >= state.pageCount) return;
        state.pageIndex = next;
        renderPageInfo();
        await loadImageForPage();
        // Preserve existing interactables but hide overlays until rescanned
        state.interactables = [];
        state.overlaysVisible = false;
        renderDebugList();
    }

    function bindEvents() {
        el.fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            try {
                await uploadPdf(file);
            } catch (err) {
                alert('Upload error: ' + err.message);
            }
        });
        el.pagePrev.addEventListener('click', () => goto(-1));
        el.pageNext.addEventListener('click', () => goto(1));
        el.detectRescan.addEventListener('click', async () => {
            try {
                // Rescan always re-fetch detections for current page/dpi
                await detectForPage();
                // Turn overlays on after a successful rescan
                state.overlaysVisible = true;
                await loadImageForPage();
                drawInteractables();
            } catch (err) {
                alert('Detect error: ' + err.message);
            }
        });
        el.detectToggle.addEventListener('click', async () => {
            // Toggle overlay visibility without discarding cached detections
            state.overlaysVisible = !state.overlaysVisible;
            await loadImageForPage();
            if (state.overlaysVisible) {
                // If we have no detections yet, fetch them first
                if (!state.interactables || state.interactables.length === 0) {
                    try {
                        await detectForPage();
                    } catch (err) {
                        alert('Detect error: ' + err.message);
                        state.overlaysVisible = false;
                        return;
                    }
                }
                drawInteractables();
            }
        });
        // Scale/DPI binding: supports numeric input or range slider if present
        if (el.scaleInput) {
            const handleScale = async () => {
                let val = el.scaleInput.value;
                if (val == null) return;
                // Accept percent (50-300), float scale (0.5-3.0), or dpi directly (72-600)
                let num = parseFloat(val);
                if (Number.isNaN(num)) return;
                // Heuristic: if <= 10, treat as scale; if <= 1000 and >= 20, treat as percent; else dpi
                if (num <= 10) {
                    state.dpi = Math.max(72, Math.min(600, Math.round(72 * num)));
                } else if (num >= 20 && num <= 1000) {
                    state.dpi = Math.max(72, Math.min(600, Math.round(1.5 * num))); // ~150 dpi at 100%
                } else {
                    state.dpi = Math.max(72, Math.min(600, Math.round(num)));
                }
                await loadImageForPage();
                // Redraw overlays if visible; bboxes are in pixel space, so they scale with image
                if (state.overlaysVisible && state.interactables && state.interactables.length) {
                    drawInteractables();
                }
            };
            el.scaleInput.addEventListener('input', handleScale);
            el.scaleInput.addEventListener('change', handleScale);
        }
        el.pageNum.addEventListener('change', async () => {
            const idx = parseInt(el.pageNum.value || '1', 10) - 1;
            if (Number.isNaN(idx)) return;
            if (idx < 0 || idx >= state.pageCount) return;
            state.pageIndex = idx;
            renderPageInfo();
            await loadImageForPage();
            state.interactables = [];
            state.overlaysVisible = false;
        });

        // Canvas click raycast: log the interactable UUID if hit
        el.canvasContainer.addEventListener('click', async (ev) => {
            if (!state.pdf || !el.canvas) return;
            // Compute coordinates relative to the canvas image pixels
            const rect = el.canvas.getBoundingClientRect();
            const scaleX = el.canvas.width / rect.width;
            const scaleY = el.canvas.height / rect.height;
            const x = Math.floor((ev.clientX - rect.left) * scaleX);
            const y = Math.floor((ev.clientY - rect.top) * scaleY);
            try {
                const url = `/raycast?pdf=${encodeURIComponent(state.pdf)}&index=${state.pageIndex}&dpi=${state.dpi}&x=${x}&y=${y}`;
                const resp = await fetch(url);
                const data = await resp.json();
                if (data && data.hit) {
                    console.log('Raycast hit:', data.hit);
                    // Debug print the UUID of the interactable
                    const id = data.hit.id || data.hit.uuid || data.hit.meta?.id;
                    console.log('Interactable UUID:', id);
                } else {
                    console.log('Raycast miss at', x, y);
                }
            } catch (err) {
                console.warn('Raycast error:', err);
            }
        });
    }

    function init() {
        ensureCanvas();
        renderPageInfo();
        bindEvents();
        // Echo front-end errors to Python backend
        window.addEventListener('error', (ev) => {
            try {
                fetch('/client-log', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        level: 'error',
                        message: ev.message || 'Unhandled error',
                        context: {
                            filename: ev.filename,
                            lineno: ev.lineno,
                            colno: ev.colno,
                            error: ev.error && (ev.error.stack || String(ev.error))
                        }
                    })
                });
            } catch { }
        });
        window.addEventListener('unhandledrejection', (ev) => {
            try {
                fetch('/client-log', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        level: 'error',
                        message: 'Unhandled promise rejection',
                        context: {
                            reason: (ev.reason && (ev.reason.stack || String(ev.reason))) || null
                        }
                    })
                });
            } catch { }
        });
    }

    document.addEventListener('DOMContentLoaded', init);
})();

