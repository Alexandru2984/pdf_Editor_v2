/* Dashboard controls use DOM text and opaque tool IDs; document data is never persisted. */
(function () {
    'use strict';

    const normalise = (value) => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase();
    const search = document.getElementById('tools-search-input');
    const grid = document.getElementById('operations-grid');
    if (!grid) return;
    const cards = Array.from(grid.querySelectorAll('.operation-card'));
    const pills = Array.from(document.querySelectorAll('[data-cat-filter]'));
    const empty = document.getElementById('tools-no-results');
    const knownIds = new Set(cards.map((card) => card.dataset.feature));
    const storageKey = 'pdfeditor-favorite-tools-v1';
    let favorites = new Set();
    let category = 'all';
    try {
        const saved = JSON.parse(localStorage.getItem(storageKey) || '[]');
        if (Array.isArray(saved)) favorites = new Set(saved.filter((id) => knownIds.has(id)));
    } catch (_) { /* Browsing and filtering work when storage is unavailable. */ }

    function filterTools() {
        const query = normalise(search.value.trim());
        let visible = 0;
        cards.forEach((card) => {
            const categories = card.dataset.category.split(/\s+/);
            const matchesCategory = category === 'all' ||
                (category === 'favorites' ? favorites.has(card.dataset.feature) : categories.includes(category));
            card.hidden = !matchesCategory || !normalise(card.dataset.searchText).includes(query);
            if (!card.hidden) visible++;
        });
        empty.hidden = visible > 0;
    }

    cards.forEach((card) => {
        // Cache searchable content before adding the favorite control/label.
        card.dataset.searchText = card.textContent + ' ' + card.dataset.feature;
        const favorite = document.createElement('button');
        favorite.type = 'button';
        favorite.className = 'tool-favorite';
        const name = card.querySelector('h3').textContent.trim();
        favorite.setAttribute('aria-label', grid.dataset.favoriteLabel + ': ' + name);
        function update() {
            const selected = favorites.has(card.dataset.feature);
            favorite.setAttribute('aria-pressed', String(selected));
            favorite.textContent = selected ? '★' : '☆';
        }
        favorite.addEventListener('click', () => {
            const id = card.dataset.feature;
            if (favorites.has(id)) favorites.delete(id);
            else favorites.add(id);
            try { localStorage.setItem(storageKey, JSON.stringify(Array.from(favorites))); } catch (_) { /* optional */ }
            update();
            filterTools();
            if (card.hidden) document.querySelector('[data-cat-filter="favorites"]').focus();
        });
        update();
        card.appendChild(favorite);
    });

    search.addEventListener('input', filterTools);
    pills.forEach((pill) => pill.addEventListener('click', () => {
        category = pill.dataset.catFilter;
        pills.forEach((item) => {
            const active = item === pill;
            item.classList.toggle('active', active);
            item.setAttribute('aria-pressed', String(active));
        });
        filterTools();
    }));
    document.addEventListener('keydown', (event) => {
        if (event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey &&
            !event.target.closest('input, textarea, select, [contenteditable="true"], [role="dialog"]')) {
            event.preventDefault();
            search.focus();
        }
    });

    document.querySelectorAll('[data-pdf-tool-jump]').forEach((select) => {
        select.addEventListener('change', () => {
            if (select.value) window.location.href = select.value + '?pdf=' + encodeURIComponent(select.dataset.pdfToolJump);
        });
    });

    const documents = document.getElementById('documents-grid');
    if (!documents) return;
    const items = Array.from(documents.querySelectorAll('[data-document-name]'));
    const documentSearch = document.getElementById('documents-search');
    const status = document.getElementById('documents-filter');
    const sort = document.getElementById('documents-sort');
    const count = document.getElementById('documents-count');
    const collator = new Intl.Collator(document.documentElement.lang, { numeric: true, sensitivity: 'base' });
    function updateDocuments() {
        const query = normalise(documentSearch.value.trim());
        let visible = 0;
        const sorted = items.slice().sort((a, b) => {
            if (sort.value === 'name') return collator.compare(a.dataset.documentName, b.dataset.documentName);
            if (sort.value === 'largest') return Number(b.dataset.documentSize) - Number(a.dataset.documentSize);
            const order = Number(b.dataset.documentDate) - Number(a.dataset.documentDate);
            return sort.value === 'oldest' ? -order : order;
        });
        sorted.forEach((item) => {
            item.hidden = !normalise(item.dataset.documentName).includes(query) ||
                (status.value !== 'all' && item.dataset.documentStatus !== status.value);
            if (!item.hidden) visible++;
            documents.appendChild(item);
        });
        count.textContent = count.dataset.countTemplate.replace('%(visible)s', String(visible)).replace('%(total)s', String(items.length));
        document.getElementById('documents-no-results').hidden = visible > 0;
    }
    documentSearch.addEventListener('input', updateDocuments);
    status.addEventListener('change', updateDocuments);
    sort.addEventListener('change', updateDocuments);
    updateDocuments();
})();
