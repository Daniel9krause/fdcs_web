// Site-wide behaviour. No inline scripts are used anywhere, so the Content-Security-Policy
// can forbid them - that blocks injected-script (XSS) attacks.
(function () {
  // mobile menu
  document.querySelectorAll("[data-menu-toggle]").forEach((btn) =>
    btn.addEventListener("click", () => document.querySelector(".nav-links").classList.toggle("open")));

  // confirmation prompts: <button data-confirm="..."> or <form data-confirm="...">
  document.addEventListener("click", (e) => {
    const el = e.target.closest("button[data-confirm]");
    if (el && !window.confirm(el.dataset.confirm)) e.preventDefault();
  });
  document.querySelectorAll("form[data-confirm]").forEach((f) =>
    f.addEventListener("submit", (e) => { if (!window.confirm(f.dataset.confirm)) e.preventDefault(); }));

  // result page: switch between annotated and original image
  const img = document.getElementById("result-img");
  document.querySelectorAll("[data-img]").forEach((b) => b.addEventListener("click", () => {
    if (img) img.src = b.dataset.img;
    document.querySelectorAll("[data-img]").forEach((x) => x.classList.toggle("btn-outline", x !== b));
  }));

  // selects that submit their form on change
  document.querySelectorAll("select[data-autosubmit]").forEach((s) =>
    s.addEventListener("change", () => s.form.submit()));
})();

// ---------------------------------------------------------------- installable web app
(function () {
  // Service workers only run on HTTPS or on this computer (localhost), not on plain http://192.168...
  if ("serviceWorker" in navigator && window.isSecureContext) {
    window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => {}));
  }
  const standalone = window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone;
  const btn = document.querySelector("[data-install]");
  let deferred = null;

  // Android / Chrome / Edge: show our own "Install app" button.
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e;
    if (btn) btn.hidden = false;
  });
  if (btn) btn.addEventListener("click", async () => {
    if (!deferred) return;
    deferred.prompt();
    await deferred.userChoice;
    deferred = null;
    btn.hidden = true;
  });
  window.addEventListener("appinstalled", () => { if (btn) btn.hidden = true; });

  // iPhone / iPad Safari has no install prompt, so show a one-time tip.
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent) && !window.MSStream;
  const hint = document.querySelector("[data-ios-hint]");
  let dismissed = false;
  try { dismissed = localStorage.getItem("fdcs-ios-hint") === "1"; } catch (e) { /* private mode */ }
  if (ios && !standalone && hint && !dismissed && window.isSecureContext) hint.hidden = false;
  document.querySelectorAll("[data-dismiss-hint]").forEach((b) => b.addEventListener("click", () => {
    hint.hidden = true;
    try { localStorage.setItem("fdcs-ios-hint", "1"); } catch (e) { /* ignore */ }
  }));
})();
