(function () {
  "use strict";

  const DOM = {
    cameraBtn:    document.getElementById("cameraBtn"),
    uploadBtn:    document.getElementById("uploadBtn"),
    cameraInput:  document.getElementById("cameraInput"),
    fileInput:    document.getElementById("fileInput"),
    preview:      document.getElementById("previewImage"),
    previewArea:  document.getElementById("previewArea"),
    placeholder:  document.getElementById("placeholder"),
    loading:      document.getElementById("loading"),
    results:      document.getElementById("results"),
    topMatch:     document.getElementById("topMatch"),
    topName:      document.getElementById("topName"),
    topProb:      document.getElementById("topProb"),
    resultsList:  document.getElementById("resultsList"),
    error:        document.getElementById("error"),
    errorMsg:     document.getElementById("errorMsg"),
    modelStatus:  document.getElementById("modelStatus"),
    infoDot:      document.querySelector(".info-dot"),
    debugContent: document.getElementById("debugContent"),
    debugLog:     document.getElementById("debugLog"),
  };

  let activeFile = null;
  const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15 MB
  const TOP_K = 5;

  // ── logging ──
  function debug(msg) {
    const el = DOM.debugContent;
    if (el) {
      const t = new Date().toLocaleTimeString("zh-CN", { hour12: false });
      el.textContent += `[${t}] ${msg}\n`;
      el.scrollTop = el.scrollHeight;
    }
  }

  // ── info dot ──
  function setStatus(state, text) {
    DOM.infoDot.className = "info-dot " + state;
    DOM.modelStatus.textContent = text;
  }

  // ── show / hide helpers ──
  function show(el) { el.classList.remove("hidden"); }
  function hide(el) { el.classList.add("hidden"); }

  function showError(msg) {
    DOM.errorMsg.textContent = msg;
    show(DOM.error);
    hide(DOM.loading);
  }

  // ── check model status ──
  async function checkModel() {
    try {
      const resp = await fetch("/api/info");
      if (!resp.ok) throw new Error("Status " + resp.status);
      const info = await resp.json();
      debug(`Model: ${info.model}, classes: ${info.num_classes}, input: ${info.input_size}x${info.input_size}`);
      setStatus("ready", "模型已就绪");
    } catch (e) {
      debug(`Model check failed: ${e.message}`);
      setStatus("error", "模型加载异常，刷新重试");
    }
  }

  // ── process image ──
  async function handleImage(file) {
    activeFile = file;

    // validate
    if (!file.type.startsWith("image/")) {
      showError("请选择图片文件");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      showError("图片超过 15MB 限制，请压缩后重试");
      return;
    }

    // preview
    const url = URL.createObjectURL(file);
    DOM.preview.src = url;
    show(DOM.preview);
    hide(DOM.placeholder);
    DOM.previewArea.classList.add("has-image");

    // clear previous
    hide(DOM.results);
    hide(DOM.error);
    hide(DOM.topMatch);
    DOM.resultsList.innerHTML = "";
    show(DOM.loading);

    debug(`Sending: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);

    try {
      const form = new FormData();
      form.append("image", file);
      form.append("top_k", TOP_K);

      const resp = await fetch("/api/predict", { method: "POST", body: form });
      const data = await resp.json();

      if (!resp.ok || data.error) {
        throw new Error(data.error || "Server error " + resp.status);
      }

      renderResults(data.results);
    } catch (e) {
      debug(`Prediction error: ${e.message}`);
      showError(e.message);
    } finally {
      hide(DOM.loading);
    }
  }

  // ── render results ──
  function renderResults(results) {
    if (!results || results.length === 0) {
      showError("未能识别出结果，请换一张图片试试");
      return;
    }

    const colors = ["#4ade80", "#34d399", "#2dd4bf", "#22d3ee", "#38bdf8"];

    // top match
    const top = results[0];
    const pct = (top.probability * 100).toFixed(1);
    DOM.topName.textContent = top.name;
    DOM.topProb.textContent = pct + "%";
    show(DOM.topMatch);

    // ranking list
    let html = "";
    results.forEach((r, i) => {
      const barPct = (r.probability * 100).toFixed(1);
      const barW = Math.max(r.probability * 100, 1);
      html += `
        <div class="result-item" style="animation-delay:${i * 0.05}s">
          <div class="result-rank">#${i + 1}</div>
          <div class="result-name">${escapeHtml(r.name)}</div>
          <div class="result-bar-wrap">
            <div class="result-bar" style="width:${barW}%;background:${colors[i % colors.length]}"></div>
          </div>
          <div class="result-pct">${barPct}%</div>
        </div>
      `;
    });
    DOM.resultsList.innerHTML = html;
    show(DOM.results);

    debug(`Results: ${results.map(r => `${r.name} ${(r.probability*100).toFixed(1)}%`).join(" | ")}`);
  }

  function escapeHtml(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  // ── camera via native capture ──
  DOM.cameraBtn.addEventListener("click", () => {
    DOM.cameraInput.value = "";
    DOM.cameraInput.click();
  });

  DOM.cameraInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImage(e.target.files[0]);
    }
  });

  // ── upload from gallery ──
  DOM.uploadBtn.addEventListener("click", () => {
    DOM.fileInput.value = "";
    DOM.fileInput.click();
  });

  DOM.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImage(e.target.files[0]);
    }
  });

  // ── drag & drop (desktop convenience) ──
  DOM.previewArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    DOM.previewArea.style.borderColor = "#4ade80";
  });
  DOM.previewArea.addEventListener("dragleave", () => {
    DOM.previewArea.style.borderColor = "";
  });
  DOM.previewArea.addEventListener("drop", (e) => {
    e.preventDefault();
    DOM.previewArea.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (file) handleImage(file);
  });

  // ── init ──
  checkModel();
  debug("App started");
})();
