const state = {
  jobId: null,
  selectedFolder: null,
  page: 1,
  pageSize: 40,
  totalItems: 0,
  items: [],
  sse: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Error de servidor");
  }
  return res.json();
}

async function refreshHealth() {
  const health = await api("/api/health");
  const container = $("health-badges");
  container.innerHTML = "";
  const makeBadge = (label, ok) => {
    const span = document.createElement("span");
    span.className = `badge ${ok ? "ok" : "warn"}`;
    span.textContent = ok ? `${label}: OK` : `${label}: Fallback`;
    return span;
  };
  container.appendChild(makeBadge("ExifTool", health.exiftool_available));
  container.appendChild(makeBadge("FFmpeg", health.ffmpeg_available));
}

function updateProgressUI(snap) {
  if (snap.job_id) state.jobId = snap.job_id;
  $("progress-stage").textContent = `Estado: ${snap.status} (${snap.stage})`;
  $("progress-stats").textContent =
    `${snap.processed_files} / ${snap.total_files} archivos · ${snap.items_per_sec} items/s · ${snap.duplicates_count} dup`;
  const pct = snap.total_files > 0 ? Math.round((snap.processed_files / snap.total_files) * 100) : 0;
  $("progress-bar").value = pct;

  $("btn-pause").disabled = snap.status !== "scanning";
  $("btn-resume").disabled = snap.status !== "paused";

  if (snap.status === "completed") {
    loadTree();
  }
}

function connectSSE() {
  if (state.sse) state.sse.close();
  const es = new EventSource("/api/events");
  es.onmessage = (evt) => {
    try {
      const snap = JSON.parse(evt.data);
      updateProgressUI(snap);
    } catch {
      // ignore malformed frames
    }
  };
  state.sse = es;
}

async function loadTree() {
  const params = new URLSearchParams({
    page: String(state.page),
    page_size: String(state.pageSize),
  });
  if (state.jobId) params.set("job_id", state.jobId);
  if (state.selectedFolder) params.set("folder", state.selectedFolder);

  const data = await api(`/api/tree?${params.toString()}`);
  state.totalItems = data.total || 0;
  state.items = data.items || [];

  const folderList = $("folder-list");
  folderList.innerHTML = "";
  for (const f of data.folders || []) {
    const li = document.createElement("li");
    li.className = `folder-item ${state.selectedFolder === f.folder ? "active" : ""}`;
    li.textContent = `${f.folder} (${f.count})`;
    li.onclick = () => {
      state.selectedFolder = f.folder;
      state.page = 1;
      loadTree();
    };
    folderList.appendChild(li);
  }

  $("current-folder-title").textContent = state.selectedFolder || "Todos los elementos";
  $("page-info").textContent = `Página ${state.page} (${state.totalItems} total)`;

  const grid = $("items-grid");
  grid.innerHTML = "";
  for (const item of state.items) {
    const card = document.createElement("div");
    card.className = "item-card";
    card.innerHTML = `
      <img src="/api/thumbnails/${encodeURIComponent(item.id)}" loading="lazy" alt="" />
      <div class="item-name">${item.target_filename}</div>
      <div class="item-sub">${item.target_folder}</div>
    `;
    card.onclick = () => showPreview(item);
    grid.appendChild(card);
  }
}

function showPreview(item) {
  const box = $("preview-box");
  box.innerHTML = `
    <img class="preview-img" src="/api/thumbnails/${encodeURIComponent(item.id)}" alt="" />
    <dl class="meta-list">
      <dt>Archivo destino</dt><dd>${item.target_filename}</dd>
      <dt>Carpeta</dt><dd>${item.target_folder}</dd>
      <dt>Fecha</dt><dd>${item.captured_at || "Sin fecha"}</dd>
      <dt>Ubicación</dt><dd>${item.location_label || "N/A"}</dd>
      <dt>Estado</dt><dd>${item.export_state}</dd>
    </dl>
  `;
}

function bindEvents() {
  $("btn-start").onclick = async () => {
    const source_dir = $("source-dir").value.trim();
    const dest_dir = $("dest-dir").value.trim();
    if (!source_dir || !dest_dir) return alert("Indica carpeta origen y destino.");
    const snap = await api("/api/scan/start", "POST", { source_dir, dest_dir });
    updateProgressUI(snap);
    connectSSE();
  };

  $("btn-pause").onclick = async () => updateProgressUI(await api("/api/scan/pause", "POST", {}));
  $("btn-resume").onclick = async () => updateProgressUI(await api("/api/scan/resume", "POST", {}));

  $("btn-prev-page").onclick = () => {
    if (state.page > 1) {
      state.page -= 1;
      loadTree();
    }
  };
  $("btn-next-page").onclick = () => {
    if (state.page * state.pageSize < state.totalItems) {
      state.page += 1;
      loadTree();
    }
  };

  $("btn-rename-folder").onclick = async () => {
    if (!state.jobId || !state.selectedFolder) return;
    const next = prompt("Nuevo nombre de carpeta:", state.selectedFolder);
    if (!next) return;
    await api("/api/tree/folders/rename", "POST", {
      job_id: state.jobId,
      old_folder: state.selectedFolder,
      new_folder: next,
    });
    state.selectedFolder = next;
    await loadTree();
  };

  $("btn-merge-folder").onclick = async () => {
    if (!state.jobId || !state.selectedFolder) return;
    const dest = prompt(`Fusionar '${state.selectedFolder}' dentro de:`, "");
    if (!dest) return;
    await api("/api/tree/folders/merge", "POST", {
      job_id: state.jobId,
      from_folder: state.selectedFolder,
      to_folder: dest,
    });
    state.selectedFolder = dest;
    await loadTree();
  };

  $("btn-create-folder").onclick = async () => {
    if (!state.jobId) return;
    const folder = prompt("Nombre de la nueva carpeta:", "2024/2024-01 - Nuevo");
    if (!folder) return;
    await api("/api/tree/folders/create", "POST", { job_id: state.jobId, folder });
    await loadTree();
  };

  $("btn-settings").onclick = async () => {
    const cfg = await api("/api/settings");
    $("cfg-exiftool").value = cfg.exiftool_path || "";
    $("cfg-ffmpeg").value = cfg.ffmpeg_path || "";
    $("cfg-gemini-key").value = cfg.gemini_api_key || "";
    $("cfg-gemini-model").value = cfg.gemini_model || "gemini-2.5-flash";
    $("settings-modal").showModal();
  };

  $("btn-close-settings").onclick = () => $("settings-modal").close();

  $("settings-form").onsubmit = async (e) => {
    e.preventDefault();
    await api("/api/settings", "POST", {
      exiftool_path: $("cfg-exiftool").value,
      ffmpeg_path: $("cfg-ffmpeg").value,
      gemini_api_key: $("cfg-gemini-key").value,
      gemini_model: $("cfg-gemini-model").value,
    });
    $("settings-modal").close();
    await refreshHealth();
  };

  $("btn-ai-classify").onclick = async () => {
    if (!state.jobId || state.items.length === 0) return;
    const ids = state.items.map((it) => it.id);
    const preview = await api("/api/ai/classify", "POST", {
      job_id: state.jobId,
      item_ids: ids,
      confirmed: false,
    });
    $("ai-confirm-text").textContent =
      `Se enviarán ${preview.item_count} elementos al modelo ${preview.model}. ¿Continuar?`;
    $("ai-confirm-modal").showModal();
  };

  $("btn-ai-cancel").onclick = () => $("ai-confirm-modal").close();
  $("btn-ai-confirm").onclick = async () => {
    const ids = state.items.map((it) => it.id);
    await api("/api/ai/classify", "POST", {
      job_id: state.jobId,
      item_ids: ids,
      confirmed: true,
    });
    $("ai-confirm-modal").close();
    await loadTree();
  };

  $("btn-export").onclick = async () => {
    if (!state.jobId) return alert("Primero ejecuta un escaneo.");
    if (!confirm("¿Aplicar y Exportar Árbol al directorio destino?")) return;
    const res = await api("/api/export", "POST", { job_id: state.jobId });
    alert(`Exportación completa: ${res.exported} exportados, ${res.duplicates} duplicados, ${res.skipped} omitidos.`);
    await loadTree();
  };
}

refreshHealth();
bindEvents();
connectSSE();
