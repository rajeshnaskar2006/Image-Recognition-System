"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const input = byId("image-input");
  const dropzone = byId("dropzone");
  const preview = byId("image-preview");
  const analyze = byId("analyze-button");
  const reset = byId("reset-button");
  const resultPanel = byId("results-panel");
  const MAX_BYTES = 10 * 1024 * 1024;
  const MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
  let selectedFile = null;
  let previewUrl = null;
  let request = null;
  let selectionVersion = 0;
  let loadingTimer = null;
  let loading = false;
  let modelReady = false;
  let dragDepth = 0;

  function announce(message) {
    byId("live-status").textContent = message;
  }

  function setError(message) {
    byId("error-message").textContent = message;
    byId("error-message").hidden = !message;
  }

  function setModelStatus(ready, unavailable = false) {
    modelReady = ready || modelReady;
    byId("status-dot").classList.toggle("ready", modelReady);
    byId("status-label").textContent = modelReady
      ? "MobileNetV2 ready"
      : unavailable ? "Server connection unavailable" : "Model loads on first analysis";
  }

  function showResultState(state) {
    byId("empty-results").hidden = state !== "empty";
    byId("loading-results").hidden = state !== "loading";
    byId("prediction-results").hidden = state !== "ready";
    byId("results-caption").textContent = state === "ready" ? "Top 5 matches" : state === "loading" ? "Analysis in progress" : "A fresh perspective";
    resultPanel.setAttribute("aria-busy", String(state === "loading"));
  }

  function setLoading(value) {
    loading = value;
    analyze.disabled = value || !selectedFile;
    analyze.classList.toggle("is-loading", value);
    byId("analyze-label").textContent = value ? "Analyzing your image…" : "Analyze image";
    clearTimeout(loadingTimer);
    if (value) {
      byId("loading-message").textContent = modelReady
        ? "Preparing your image and finding its most likely match."
        : "Loading the model and preparing your first prediction.";
      loadingTimer = setTimeout(() => {
        byId("loading-message").textContent = "Still working. Loading the model for the first time can take a few minutes.";
      }, 15000);
    }
  }

  function cancelCurrentRequest() {
    selectionVersion += 1;
    if (request) request.abort();
    request = null;
    setLoading(false);
  }

  function discardPreview() {
    preview.onload = null;
    preview.onerror = null;
    preview.removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  function clearSelection() {
    cancelCurrentRequest();
    discardPreview();
    selectedFile = null;
    input.value = "";
    preview.hidden = true;
    byId("empty-upload").hidden = false;
    byId("preview-change").hidden = true;
    byId("file-details").hidden = true;
    reset.hidden = true;
    analyze.disabled = true;
    dropzone.classList.remove("has-image", "dragover");
    dropzone.setAttribute("aria-label", "Choose an image, or drop one here");
    showResultState("empty");
    setError("");
    announce("Image cleared. Choose an image to begin.");
  }

  function selectFile(file) {
    if (!file) return;
    // Changing a file invalidates both pending decoding and any older network response.
    clearSelection();
    const version = selectionVersion;
    const validType = MIME_TYPES.has(file.type) || (!file.type && /\.(jpe?g|png|webp)$/i.test(file.name));
    if (!validType) {
      setError("Choose a JPG, PNG or WebP image. Other file types are not supported.");
      return;
    }
    if (file.size === 0) {
      setError("This file is empty. Please choose another image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError("This image is larger than 10 MB. Choose a smaller image and try again.");
      return;
    }
    previewUrl = URL.createObjectURL(file);
    preview.onload = () => {
      if (version !== selectionVersion) return;
      selectedFile = file;
      preview.hidden = false;
      byId("empty-upload").hidden = true;
      byId("preview-change").hidden = false;
      byId("file-details").hidden = false;
      byId("file-name").textContent = file.name;
      byId("file-name").title = file.name;
      const fileSize = file.size >= 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
      byId("file-size").textContent = `${fileSize} · ${preview.naturalWidth} × ${preview.naturalHeight} px`;
      dropzone.classList.add("has-image");
      dropzone.setAttribute("aria-label", "Change selected image");
      reset.hidden = false;
      analyze.disabled = false;
      announce(`${file.name} selected. Ready to analyze.`);
    };
    preview.onerror = () => {
      if (version !== selectionVersion) return;
      clearSelection();
      setError("This image could not be opened. It may be damaged or use an unsupported format.");
    };
    preview.src = previewUrl;
  }

  function formatLabel(label) {
    const words = String(label).replace(/_/g, " ");
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  function renderResults(data) {
    if (!Array.isArray(data.predictions) || data.predictions.length === 0) throw new Error("The server returned no predictions. Please try again.");
    const predictions = data.predictions.slice(0, 5).map((item) => {
      const confidence = item.confidence;
      if (typeof item.label !== "string" || !item.label.trim() || typeof confidence !== "number" || !Number.isFinite(confidence) || confidence < 0 || confidence > 100) {
        throw new Error("The server returned an invalid prediction. Please try again.");
      }
      return {label: formatLabel(item.label), confidence};
    });
    predictions.sort((a, b) => b.confidence - a.confidence);
    const best = predictions[0];
    byId("best-label").textContent = best.label;
    byId("best-confidence").textContent = `${best.confidence.toFixed(1)}%`;
    const list = byId("prediction-list");
    list.replaceChildren();
    predictions.forEach((prediction, index) => {
      const row = document.createElement("li");
      row.className = "prediction-row";
      const rank = document.createElement("span");
      rank.className = "rank-number";
      rank.textContent = String(index + 1).padStart(2, "0");
      rank.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "prediction-name";
      name.textContent = prediction.label;
      const percentage = document.createElement("span");
      percentage.className = "prediction-percent";
      percentage.textContent = `${prediction.confidence.toFixed(1)}%`;
      const track = document.createElement("div");
      track.className = "prediction-track";
      track.setAttribute("aria-hidden", "true");
      const bar = document.createElement("div");
      bar.className = "prediction-bar";
      bar.style.width = `${prediction.confidence}%`;
      track.appendChild(bar);
      row.append(rank, name, percentage, track);
      list.appendChild(row);
    });
    byId("confidence-note").hidden = !data.low_confidence;
    const elapsed = data.elapsed_ms;
    byId("elapsed-time").textContent = typeof elapsed === "number" && Number.isFinite(elapsed) && elapsed >= 0
      ? elapsed < 1000 ? `${Math.round(elapsed)} ms` : `${(elapsed / 1000).toFixed(1)} s`
      : "Analysis complete";
    const dimensions = data.image;
    byId("image-dimensions").textContent = dimensions && Number.isInteger(dimensions.width) && Number.isInteger(dimensions.height) && dimensions.width > 0 && dimensions.height > 0
      ? `${dimensions.width} × ${dimensions.height} px` : "";
    showResultState("ready");
    byId("results-caption").textContent = `Top ${predictions.length} matches`;
    announce(`Analysis complete. Most likely match: ${best.label}, ${best.confidence.toFixed(1)} percent model confidence.${data.low_confidence ? " The model is uncertain about this image." : ""}`);
    setModelStatus(true);
  }

  async function classifyImage() {
    if (!selectedFile || loading) return;
    setError("");
    const version = selectionVersion;
    const controller = new AbortController();
    request = controller;
    setLoading(true);
    showResultState("loading");
    announce("Analyzing image. The first analysis may take longer while the model loads.");
    const body = new FormData();
    body.append("image", selectedFile);
    try {
      const response = await fetch("/api/predict", {method: "POST", body, signal: controller.signal, headers: {Accept: "application/json"}});
      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error(response.status === 413 ? "This upload is too large. Please choose an image under 10 MB." : "The server returned an unexpected response. Please try again.");
      }
      if (version !== selectionVersion) return;
      if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : "The image could not be analyzed. Please try again.");
      renderResults(data);
    } catch (error) {
      if (version !== selectionVersion || error.name === "AbortError") return;
      showResultState("empty");
      const message = error instanceof TypeError ? "Could not reach the server. Check that the app is running, then try again." : error.message || "Something went wrong. Please try again.";
      setError(message);
      announce("Analysis failed. " + message);
    } finally {
      if (version === selectionVersion) {
        request = null;
        setLoading(false);
      }
    }
  }

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => selectFile(input.files[0]));
  dropzone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  dropzone.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    dropzone.classList.remove("dragover");
    const files = event.dataTransfer ? event.dataTransfer.files : [];
    if (files.length > 1) {
      setError("Please choose one image at a time.");
      return;
    }
    selectFile(files[0]);
  });
  // Prevent dropping a file outside the picker from navigating away from the app.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  analyze.addEventListener("click", classifyImage);
  reset.addEventListener("click", () => {
    clearSelection();
    dropzone.focus();
  });
  window.addEventListener("beforeunload", () => {
    if (request) request.abort();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });

  async function checkHealth() {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch("/api/health", {signal: controller.signal, headers: {Accept: "application/json"}});
      if (!response.ok) throw new Error("Health check failed");
      const data = await response.json();
      setModelStatus(data.model_ready === true, data.status !== "ok");
    } catch {
      setModelStatus(false, true);
    } finally {
      clearTimeout(timeout);
    }
  }
  checkHealth();
})();
