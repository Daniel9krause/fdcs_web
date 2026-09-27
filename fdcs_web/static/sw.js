/* FDCS service worker - makes the site installable and usable with a weak connection.
 *
 * - App shell (CSS, JS, icons) is cached, so pages open fast on slow mobile data.
 * - Pages always try the network first (prices, results and orders stay current),
 *   and fall back to the last saved copy, or an offline page, when there's no signal.
 * - Never caches: form submissions, the owner area, orders, scan results or photos
 *   (those are private and must not be stored on a shared phone).
 */
const VERSION = "fdcs-v1";
const SHELL = [
  "/offline",
  "/static/css/style.css",
  "/static/js/app.js",
  "/static/js/scan.js",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/manifest.webmanifest",
];
const PUBLIC_PAGES = ["/", "/scan", "/pests", "/store", "/about", "/contact"];
const PRIVATE = /^\/(admin|order|result|history|gallery|uploads|api|cart|checkout|payment|country)/;

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function isPublicPage(path) {
  return PUBLIC_PAGES.includes(path) || path.startsWith("/pests/") || path.startsWith("/store/");
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin || PRIVATE.test(url.pathname)) {
    return; // let the browser handle it normally
  }

  // Static files: cache first, refresh in the background.
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(
      caches.open(VERSION).then(async (cache) => {
        const hit = await cache.match(req);
        const fresh = fetch(req).then((res) => { if (res.ok) cache.put(req, res.clone()); return res; })
          .catch(() => hit);
        return hit || fresh;
      })
    );
    return;
  }

  // Pages: network first, then the saved copy, then the offline page.
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const noStore = (res.headers.get("Cache-Control") || "").includes("no-store");
          if (res.ok && !noStore && isPublicPage(url.pathname)) {
            const copy = res.clone();
            caches.open(VERSION).then((c) => c.put(req, copy));
          }
          return res;
        })
        .catch(async () => (await caches.match(req)) || caches.match("/offline"))
    );
  }
});
