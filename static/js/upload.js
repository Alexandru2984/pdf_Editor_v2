/* Shared upload preflight for the dashboard and the dedicated upload form. */
(function () {
    'use strict';

    document.querySelectorAll('[data-pdf-upload]').forEach((form) => {
        const input = form.querySelector('input[type="file"]');
        const feedback = form.querySelector('[data-upload-feedback]');
        if (!input || !feedback) return;
        const error = feedback.querySelector('[role="alert"]');
        const status = feedback.querySelector('[data-upload-status]');
        const zone = form.closest('#drop-zone');
        const fileName = form.querySelector('#fileName');
        const emptyLabel = fileName ? fileName.textContent : '';
        let submitting = false;

        function validate() {
            const files = Array.from(input.files || []);
            let message = '';
            if (files.some((file) => !file.name.toLowerCase().endsWith('.pdf'))) {
                message = feedback.dataset.typeError;
            } else if (files.length > Number(feedback.dataset.maxFiles)) {
                message = feedback.dataset.countError;
            } else if (files.some((file) => file.size > Number(feedback.dataset.maxFileBytes))) {
                message = feedback.dataset.sizeError;
            } else if (files.reduce((sum, file) => sum + file.size, 0) > Number(feedback.dataset.maxBatchBytes)) {
                message = feedback.dataset.batchError;
            }
            error.textContent = message;
            error.hidden = !message;
            input.setAttribute('aria-invalid', String(Boolean(message)));
            if (fileName) fileName.textContent = files.length ? files.map((file) => file.name).join(', ') : emptyLabel;
            return files.length > 0 && !message;
        }

        input.addEventListener('change', () => {
            if (submitting) return;
            if (validate() && form.hasAttribute('data-auto-upload')) form.requestSubmit();
        });

        form.addEventListener('submit', (event) => {
            if (submitting || !validate()) {
                event.preventDefault();
                return;
            }
            submitting = true;
            form.setAttribute('aria-busy', 'true');
            form.classList.add('upload-busy');
            status.textContent = feedback.dataset.uploading;
            form.querySelectorAll('button[type="submit"]').forEach((button) => { button.disabled = true; });
        });

        // Restore controls when Back returns a page from the browser's page cache.
        window.addEventListener('pageshow', () => {
            submitting = false;
            form.removeAttribute('aria-busy');
            form.classList.remove('upload-busy');
            status.textContent = '';
            form.querySelectorAll('button[type="submit"]').forEach((button) => { button.disabled = false; });
        });

        if (!zone) return;
        zone.addEventListener('click', (event) => {
            if (!submitting && !event.target.closest('label, input, button, a')) input.click();
        });
        zone.addEventListener('dragover', (event) => {
            event.preventDefault();
            if (!submitting) zone.classList.add('drag-over');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
        zone.addEventListener('drop', (event) => {
            event.preventDefault();
            zone.classList.remove('drag-over');
            if (submitting || !event.dataTransfer || !event.dataTransfer.files.length) return;
            input.files = event.dataTransfer.files;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        });
    });
})();
