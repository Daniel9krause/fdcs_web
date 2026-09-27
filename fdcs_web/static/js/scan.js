// Camera capture + upload handling for the scan page.
(function () {
  const $ = (id) => document.getElementById(id);
  const video = $("video"), preview = $("preview"), placeholder = $("placeholder");
  const canvas = $("canvas"), cameraField = $("camera_image"), fileInput = $("file");
  const analyse = $("analyse"), snap = $("snap"), startCam = $("start-cam");
  const switchCam = $("switch-cam"), retake = $("retake"), dropzone = $("dropzone");
  let stream = null, facing = "environment";

  function show(el, on) { el.hidden = !on; }
  function ready(on) { analyse.disabled = !on; }

  function showPreview(src) {
    preview.src = src; show(preview, true); show(video, false); show(placeholder, false); ready(true);
  }

  function stopCamera() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    snap.disabled = true; switchCam.disabled = true;
  }

  async function startCamera() {
    stopCamera();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Live camera needs HTTPS or localhost. Use the Upload tab - on a phone it can open the camera.");
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
      });
      video.srcObject = stream;
      show(video, true); show(preview, false); show(placeholder, false); show(retake, false);
      snap.disabled = false; switchCam.disabled = false; cameraField.value = ""; ready(false);
    } catch (err) {
      alert("Could not open the camera: " + err.message + "\nYou can still upload a photo.");
    }
  }

  function capture() {
    if (!stream) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const data = canvas.toDataURL("image/jpeg", 0.9);
    cameraField.value = data; fileInput.value = "";
    stopCamera(); showPreview(data); show(retake, true);
  }

  function useFile(file) {
    if (!file || !file.type.startsWith("image/")) { alert("Please choose an image file."); return; }
    if (file.size > 12 * 1024 * 1024) { alert("Image is larger than 12 MB."); return; }
    cameraField.value = "";
    const reader = new FileReader();
    reader.onload = (e) => showPreview(e.target.result);
    reader.readAsDataURL(file);
  }

  // tabs
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    const cam = tab.dataset.mode === "camera";
    show($("camera-controls"), cam); show($("upload-controls"), !cam);
    if (!cam) stopCamera(); show(video, false);
    if (!preview.src || preview.hidden) { show(placeholder, true); ready(false); }
  }));

  startCam.addEventListener("click", startCamera);
  snap.addEventListener("click", capture);
  retake.addEventListener("click", startCamera);
  switchCam.addEventListener("click", () => { facing = facing === "environment" ? "user" : "environment"; startCamera(); });
  fileInput.addEventListener("change", () => useFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) { const dt = new DataTransfer(); dt.items.add(f); fileInput.files = dt.files; useFile(f); }
  });

  $("scan-form").addEventListener("submit", (e) => {
    if (!cameraField.value && !fileInput.files.length) { e.preventDefault(); alert("Take or choose a photo first."); return; }
    $("loading").classList.add("show"); analyse.disabled = true; stopCamera();
  });
  window.addEventListener("pagehide", stopCamera);
})();
