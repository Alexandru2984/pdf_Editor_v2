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

// Bump on every deploy so clients with a stale SW pick up new assets.
const CACHE_NAME = "pdfeditor-shell-v4";
const APP_SHELL = [
    "/static/offline.html",
    "/static/css/style.css",
    "/static/css/pdf-modal.css",
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

    // Never intercept API, admin, accounts, downloads, or streaming routes.
    const NO_CACHE_PREFIXES = [
        "/api/",
        "/admin/",
        "/accounts/",
        "/media/",
        "/download_",
        "/jobs/",
        "/chat/",
        "/s/",
    ];
    if (NO_CACHE_PREFIXES.some((p) => url.pathname.startsWith(p))) return;

    // Cache immutable/static assets cache-first.
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

    // For HTML navigations, never persist dynamic pages in CacheStorage
    // (they carry per-request CSP nonces, CSRF tokens, and session state).
    // Hit the network and fall back to the offline shell page if unreachable.
    if (req.mode === "navigate") {
        event.respondWith(
            fetch(req).catch(() => caches.match("/static/offline.html"))
        );
    }
});

// Clicking a notification should re-focus an existing tab on the matching
// URL (the typical case: user opened a job page, switched tabs, got
// notified, clicks). Falls back to opening a new tab.
self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    const target = (event.notification.data && event.notification.data.url) || "/";
    event.waitUntil(
        self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clientsList) => {
            for (const client of clientsList) {
                try {
                    const u = new URL(client.url);
                    if (u.pathname === target && "focus" in client) {
                        return client.focus();
                    }
                } catch (_) { /* ignore */ }
            }
            return self.clients.openWindow(target);
        })
    );
});
