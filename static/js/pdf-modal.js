/* PDF preview modal.
 *
 * PDF.js is deliberately loaded only after a user asks for a preview. The
 * library + worker are over 1 MiB uncompressed and most public visits never
 * use them. Asset URLs come from Django's static tag, so production manifest
 * hashing continues to work.
 */
(function () {
    "use strict";

    const loaderElement = document.currentScript;
    const pdfJsUrl = loaderElement?.dataset.pdfjsUrl || "";
    const pdfJsWorkerUrl = loaderElement?.dataset.pdfjsWorkerUrl || "";
    const translate = window.gettext || ((value) => value);

    class PDFPreviewModal {
        constructor() {
            this.modal = null;
            this.pdfDoc = null;
            this.pageNum = 1;
            this.scale = 1.5;
            this.canvas = null;
            this.ctx = null;
            this.pageRendering = false;
            this.pageNumPending = null;
            this.pdfJsPromise = null;
            this.init();
        }

        init() {
            const modalHTML = `
                <div id="pdf-preview-modal" class="pdf-modal" role="dialog" aria-modal="true" aria-labelledby="pdf-modal-title">
                    <div class="pdf-modal-content">
                        <div class="pdf-modal-header">
                            <h3 id="pdf-modal-title">📄 ${translate("PDF Preview")}</h3>
                            <button type="button" class="pdf-modal-close" aria-label="${translate("Close")}">&times;</button>
                        </div>
                        <div class="pdf-modal-body">
                            <div class="pdf-canvas-wrapper">
                                <canvas id="modal-pdf-canvas"></canvas>
                            </div>
                            <div class="pdf-modal-controls">
                                <button type="button" id="modal-prev-page" class="icon-btn">${translate("← Previous")}</button>
                                <span class="page-info">
                                    ${translate("Page")} <span id="modal-page-num"></span> ${translate("of")} <span id="modal-page-count"></span>
                                </span>
                                <button type="button" id="modal-next-page" class="icon-btn">${translate("Next →")}</button>
                                <div class="zoom-controls">
                                    <button type="button" id="modal-zoom-out" class="icon-btn" aria-label="${translate("Zoom out")}">−</button>
                                    <span class="page-info"><span id="modal-zoom-level">100</span>%</span>
                                    <button type="button" id="modal-zoom-in" class="icon-btn" aria-label="${translate("Zoom in")}">+</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML("beforeend", modalHTML);

            this.modal = document.getElementById("pdf-preview-modal");
            this.canvas = document.getElementById("modal-pdf-canvas");
            this.ctx = this.canvas.getContext("2d");
            this.setupEventListeners();
        }

        setupEventListeners() {
            this.modal.addEventListener("click", (event) => {
                if (event.target === this.modal) this.close();
            });
            this.modal.querySelector(".pdf-modal-close").addEventListener("click", () => this.close());

            document.getElementById("modal-prev-page").addEventListener("click", () => {
                if (this.pageNum <= 1) return;
                this.pageNum -= 1;
                this.queueRenderPage(this.pageNum);
            });
            document.getElementById("modal-next-page").addEventListener("click", () => {
                if (this.pageNum >= this.pdfDoc.numPages) return;
                this.pageNum += 1;
                this.queueRenderPage(this.pageNum);
            });
            document.getElementById("modal-zoom-in").addEventListener("click", () => {
                this.scale += 0.25;
                document.getElementById("modal-zoom-level").textContent = Math.round(this.scale * 100);
                this.queueRenderPage(this.pageNum);
            });
            document.getElementById("modal-zoom-out").addEventListener("click", () => {
                if (this.scale <= 0.5) return;
                this.scale -= 0.25;
                document.getElementById("modal-zoom-level").textContent = Math.round(this.scale * 100);
                this.queueRenderPage(this.pageNum);
            });
            document.addEventListener("keydown", (event) => {
                if (!this.modal.classList.contains("active")) return;
                if (event.key === "Escape") this.close();
                if (event.key === "ArrowLeft") document.getElementById("modal-prev-page").click();
                if (event.key === "ArrowRight") document.getElementById("modal-next-page").click();
            });
        }

        async ensurePdfJs() {
            if (window.pdfjsLib) return window.pdfjsLib;
            if (this.pdfJsPromise) return this.pdfJsPromise;
            if (!pdfJsUrl || !pdfJsWorkerUrl) throw new Error("PDF.js asset URLs are missing");

            this.pdfJsPromise = new Promise((resolve, reject) => {
                const script = document.createElement("script");
                script.src = pdfJsUrl;
                script.async = true;
                script.addEventListener("load", () => {
                    window.pdfjsLib.GlobalWorkerOptions.workerSrc = pdfJsWorkerUrl;
                    resolve(window.pdfjsLib);
                });
                script.addEventListener("error", () => reject(new Error("PDF.js failed to load")));
                document.head.appendChild(script);
            });
            return this.pdfJsPromise;
        }

        async open(pdfUrl) {
            try {
                await this.ensurePdfJs();
                const normalizedUrl =
                    pdfUrl.startsWith("/") && !pdfUrl.startsWith("//")
                        ? window.location.origin + pdfUrl
                        : pdfUrl;
                const loadingTask = window.pdfjsLib.getDocument({
                    url: normalizedUrl,
                    withCredentials: false,
                    isEvalSupported: false,
                });

                this.pdfDoc = await loadingTask.promise;
                this.pageNum = 1;
                this.scale = 1.5;
                document.getElementById("modal-page-count").textContent = this.pdfDoc.numPages;
                document.getElementById("modal-zoom-level").textContent = Math.round(this.scale * 100);
                this.modal.classList.add("active");
                document.body.style.overflow = "hidden";
                await this.renderPage(this.pageNum);
            } catch (error) {
                console.error("Error loading PDF preview:", error);
                alert(`${translate("Error loading PDF preview")}: ${error.message}`);
            }
        }

        close() {
            this.modal.classList.remove("active");
            document.body.style.overflow = "";
            if (this.pdfDoc) {
                this.pdfDoc.destroy();
                this.pdfDoc = null;
            }
        }

        async renderPage(number) {
            this.pageRendering = true;
            const page = await this.pdfDoc.getPage(number);
            const viewport = page.getViewport({ scale: this.scale });
            this.canvas.height = viewport.height;
            this.canvas.width = viewport.width;
            await page.render({ canvasContext: this.ctx, viewport }).promise;
            this.pageRendering = false;
            if (this.pageNumPending !== null) {
                const pending = this.pageNumPending;
                this.pageNumPending = null;
                await this.renderPage(pending);
            }
            document.getElementById("modal-page-num").textContent = number;
            this.updateButtons();
        }

        queueRenderPage(number) {
            if (this.pageRendering) {
                this.pageNumPending = number;
            } else {
                void this.renderPage(number);
            }
        }

        updateButtons() {
            document.getElementById("modal-prev-page").disabled = this.pageNum <= 1;
            document.getElementById("modal-next-page").disabled = this.pageNum >= this.pdfDoc.numPages;
        }
    }

    function initialize() {
        if (document.querySelector("[data-pdf-modal]")) {
            window.pdfModal = new PDFPreviewModal();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})();
