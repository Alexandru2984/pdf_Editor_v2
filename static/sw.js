/* PDF Editor service worker.
 *
 * Conservative caching strategy — this app is content-heavy and per-user,
 * so we don't want to serve stale state. We only pre-cache the bare app
 * shell (CSS + the modal viewer JS + logo + offline fallback), and for
 * everything else we go network-first with a same-origin cache fallback
 * so the UI still works offline once a page has been visited.
 *
 * Skip caching for: API routes, anything POST-y, and per-user pages
 * (which depend on session cookies).
 */

const CACHE_NAME = "pdfeditor-shell-v1";
const APP_SHELL = [
    "/static/css/style.css",
    "/static/css/pdf-modal.css",
    "/static/css/chat.css",
    "/static/js/pdf-modal.js",
    "/static/images/logo.jpg",
    "/static/offline.html",
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
    );
    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
            )
        )
    );
    self.clients.claim();
});

self.addEventListener("fetch", (event) => {
    const req = event.request;
    if (req.method !== "GET") return;

    const url = new URL(req.url);
    if (url.origin !== self.location.origin) return;

    // Never cache API, admin, downloads, or anything that depends on the
    // current session/user. Those must always hit the network so we don't
    // serve stale per-user data.
    const NO_CACHE_PREFIXES = ["/api/", "/admin/", "/media/", "/download_", "/jobs/", "/chat/"];
    if (NO_CACHE_PREFIXES.some((p) => url.pathname.startsWith(p))) return;

    // Cache the app shell (static assets) cache-first.
    if (url.pathname.startsWith("/static/")) {
        event.respondWith(
            caches.match(req).then((hit) => hit || fetch(req).then((resp) => {
                if (resp.ok) {
                    const copy = resp.clone();
                    caches.open(CACHE_NAME).then((c) => c.put(req, copy));
                }
                return resp;
            }))
        );
        return;
    }

    // For navigations, try the network first; fall back to last-cached
    // version if offline; if neither, show the offline page.
    if (req.mode === "navigate") {
        event.respondWith(
            fetch(req)
                .then((resp) => {
                    if (resp.ok) {
                        const copy = resp.clone();
                        caches.open(CACHE_NAME).then((c) => c.put(req, copy));
                    }
                    return resp;
                })
                .catch(() => caches.match(req).then((cached) => cached || caches.match("/static/offline.html")))
        );
    }
});
