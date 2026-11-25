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

const PHASE_MAP = {
  init: "1/4 Depurando duplicados",
  discovery: "1/4 Depurando duplicados",
  hashing: "1/4 Depurando duplicados",
  metadata: "2/4 Clasificando fotos y capturas",
  clustering: "3/4 Nombrando carpetas con IA",
  completed: "4/4 Listo",
  partial: "4/4 Listo",
};

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

  const aiModel = health.ai_model || health.model || "gemini-4-hydrogen-medium";
  if (health.agy_ok) {
    const aiPill = document.createElement("span");
    aiPill.className = "badge ok";
    aiPill.textContent = `IA: agy OK · ${aiModel}`;
    container.appendChild(aiPill);
  } else {
    const aiBtn = document.createElement("button");
    aiBtn.type = "button";
    aiBtn.className = "badge warn badge-link";
    aiBtn.title = "Abrir Configuración para ajustar Antigravity CLI";
    aiBtn.textContent = "IA: Fallback local (agy no encontrado)";
    aiBtn.onclick = () => $("btn-settings")?.click();
    container.appendChild(aiBtn);
  }
}

function logActivity(message) {
  const list = $("activity-log");
  if (!list) return;
  const now = new Date().toLocaleTimeString("es-MX", { hour12: false });
  const li = document.createElement("li");
  li.textContent = `[${now}] ${message}`;
  list.prepend(li);
  while (list.children.length > 60) {
    list.removeChild(list.lastChild);
  }
}

function showToast(message, variant = "ok") {
  const toast = $("result-toast");
  if (!toast) return;
  toast.textContent = message;
  toast.className = `toast-banner ${variant}`;
}

function setActionBusy(activeBtnId, busyText) {
  const actionIds = ["btn-start", "btn-ai-regenerate", "btn-export"];
  const originals = {};
  for (const id of actionIds) {
    const el = $(id);
    if (!el) continue;
    originals[id] = el.textContent;
    el.disabled = true;
    if (id === activeBtnId && busyText) {
      el.innerHTML = `<span class="spinner" aria-hidden="true"></span> ${busyText}`;
    }
  }
  return () => {
    for (const id of actionIds) {
      const el = $(id);
      if (!el) continue;
      el.disabled = false;
      if (originals[id] !== undefined) {
        el.textContent = originals[id];
      }
    }
  };
}

function getBatchOptions() {
  const sizeEl = $("batch-size");
  const soloMain = $("solo-primer-lote");
  const soloAdv = $("chk-solo-lote-adv");
  const solo = Boolean((soloMain && soloMain.checked) || (soloAdv && soloAdv.checked));
  return {
    batch_size: sizeEl ? Number(sizeEl.value || 100) : 100,
    solo_primer_lote: solo,
  };
}

function renderBatches(batches) {
  const listEl = $("batch-list");
  const countEl = $("batch-summary-count");
  if (!listEl || !countEl) return;

  const items = Array.isArray(batches) ? batches : [];
  countEl.textContent = `${items.length} ${items.length === 1 ? "lote" : "lotes"}`;
  listEl.innerHTML = "";

  if (items.length === 0) {
    return;
  }

  const stateLabels = {
    pendiente: "pendiente",
    en_proceso: "en_proceso",
    completado: "completado",
    error: "error",
  };

  items.forEach((b, idx) => {
    const totalBatches = b.total_batches || items.length;
    const batchNum = b.batch_index || idx + 1;
    const status = stateLabels[b.status] || b.status || "pendiente";
    const total = Number(b.total_items || 0);
    const processed = Number(b.processed_items || 0);
    const dups = Number(b.duplicates_skipped || 0);
    const errors = Number(b.errors || 0);
    const elapsed = Number(b.elapsed_seconds || 0).toFixed(2);

    const card = document.createElement("div");
    card.className = `batch-card batch-status-${status}`;

    const header = document.createElement("div");
    header.className = "batch-card-header";

    const title = document.createElement("strong");
    title.textContent = `Lote ${batchNum} de ${totalBatches}`;

    const badge = document.createElement("span");
    badge.className = `badge batch-badge ${status}`;
    badge.textContent = status;

    header.appendChild(title);
    header.appendChild(badge);

    const prog = document.createElement("progress");
    prog.max = total > 0 ? total : 1;
    prog.value = processed;

    const meta = document.createElement("div");
    meta.className = "batch-card-meta";
    meta.textContent = `${processed} / ${total} elementos · ${dups} dup · ${elapsed}s${errors > 0 ? ` · ${errors} errores` : ""}`;

    card.appendChild(header);
    card.appendChild(prog);
    card.appendChild(meta);

    if (b.error_message) {
      const errEl = document.createElement("div");
      errEl.className = "batch-card-error";
      errEl.textContent = b.error_message;
      card.appendChild(errEl);
    }

    listEl.appendChild(card);
  });
}

function updateProgressUI(snap) {
  if (snap.job_id) state.jobId = snap.job_id;
  const phaseLabel = snap.phase_label || PHASE_MAP[snap.stage] || "1/4 Depurando duplicados";
  if ($("progress-phase")) $("progress-phase").textContent = phaseLabel;
  $("progress-stage").textContent = `Estado: ${snap.status}`;
  $("progress-stats").textContent =
    `${snap.processed_files} / ${snap.total_files} archivos · ${snap.items_per_sec} items/s`;

  const pct = snap.total_files > 0 ? Math.round((snap.processed_files / snap.total_files) * 100) : 0;
  $("progress-bar").value = snap.status === "completed" ? 100 : pct;

  const newCount = snap.new_count ?? Math.max(0, (snap.processed_files || 0) - (snap.duplicates_count || 0));
  if ($("cnt-new")) $("cnt-new").textContent = String(newCount);
  if ($("cnt-dups")) $("cnt-dups").textContent = String(snap.duplicates_count || 0);
  if ($("cnt-screenshots")) $("cnt-screenshots").textContent = String(snap.screenshots_count || 0);

  const activePhaseNum = Number(phaseLabel[0]) || 1;
  document.querySelectorAll(".phase-step").forEach((el) => {
    const pNum = Number(el.getAttribute("data-phase") || 0);
    el.classList.toggle("active", pNum === activePhaseNum);
    el.classList.toggle("done", pNum < activePhaseNum || snap.status === "completed");
  });

  if ($("btn-pause")) $("btn-pause").disabled = snap.status !== "scanning";
  if ($("btn-resume")) $("btn-resume").disabled = snap.status !== "paused";

  if (Array.isArray(snap.batches) && snap.batches.length > 0) {
    renderBatches(snap.batches);
  }

  if (snap.status === "completed" || snap.status === "partial") {
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
    const isActive = state.selectedFolder === f.folder;
    li.className = `folder-item ${isActive ? "active" : ""}`;
    li.dataset.folder = f.folder;
    li.setAttribute("role", "treeitem");
    li.setAttribute("aria-selected", isActive ? "true" : "false");
    li.tabIndex = 0;
    li.textContent = `${f.folder} (${f.count})`;
    const selectFolder = () => {
      state.selectedFolder = f.folder;
      state.page = 1;
      loadTree();
    };
    li.onclick = selectFolder;
    li.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectFolder();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        if (li.nextElementSibling) {
          li.nextElementSibling.focus();
        }
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (li.previousElementSibling) {
          li.previousElementSibling.focus();
        }
      }
    };
    folderList.appendChild(li);
  }

  $("current-folder-title").textContent = state.selectedFolder || "Todos los elementos";
  $("page-info").textContent = `Página ${state.page} (${state.totalItems} total)`;

  const grid = $("items-grid");
  grid.innerHTML = "";
  if (state.items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    const h3 = document.createElement("h3");
    h3.textContent = state.jobId
      ? "Esta carpeta está vacía"
      : "Aún no hay elementos escaneados";
    const p = document.createElement("p");
    p.textContent = state.jobId
      ? "Selecciona otra carpeta en el árbol izquierdo para ver sus elementos."
      : "Selecciona tu Carpeta con fotos nuevas y Mi Biblioteca Organizada arriba, y pulsa Organizar Automáticamente.";
    empty.appendChild(h3);
    empty.appendChild(p);
    grid.appendChild(empty);
    return;
  }
  for (const item of state.items) {
    const card = document.createElement("div");
    card.className = "item-card";
    card.setAttribute("role", "listitem");
    card.tabIndex = 0;

    const img = document.createElement("img");
    img.src = `/api/thumbnails/${encodeURIComponent(item.id)}`;
    img.loading = "lazy";
    img.alt = item.target_filename || "";

    const nameEl = document.createElement("div");
    nameEl.className = "item-name";
    nameEl.textContent = item.target_filename || "";

    const subEl = document.createElement("div");
    subEl.className = "item-sub";
    subEl.textContent = item.target_folder || "";

    card.appendChild(img);
    card.appendChild(nameEl);
    card.appendChild(subEl);
    card.onclick = () => showPreview(item);
    card.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        showPreview(item);
      }
    };
    grid.appendChild(card);
  }
}

function showPreview(item) {
  const box = $("preview-box");
  box.innerHTML = "";

  const img = document.createElement("img");
  img.className = "preview-img";
  img.src = `/api/thumbnails/${encodeURIComponent(item.id)}`;
  img.alt = item.target_filename || "Vista previa";

  const dl = document.createElement("dl");
  dl.className = "meta-list";
  const fields = [
    ["Archivo destino", item.target_filename || ""],
    ["Carpeta", item.target_folder || ""],
    ["Fecha", item.captured_at || "Sin fecha"],
    ["Ubicación", item.location_label || "N/A"],
    ["Estado", item.export_state || ""],
  ];
  for (const [label, value] of fields) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.appendChild(dt);
    dl.appendChild(dd);
  }

  box.appendChild(img);
  box.appendChild(dl);
}

function bindEvents() {
  const themeBtn = $("btn-theme");
  if (themeBtn) {
    themeBtn.onclick = () => {
      const root = document.documentElement;
      const isLight = root.getAttribute("data-theme") === "light";
      if (isLight) {
        root.removeAttribute("data-theme");
      } else {
        root.setAttribute("data-theme", "light");
      }
    };
  }

  const syncChecks = (src, dst) => {
    if (src && dst) {
      src.onchange = () => {
        dst.checked = src.checked;
      };
    }
  };
  syncChecks($("solo-primer-lote"), $("chk-solo-lote-adv"));
  syncChecks($("chk-solo-lote-adv"), $("solo-primer-lote"));

  const browseFolder = async (inputId) => {
    const input = $(inputId);
    const res = await api("/api/browse-folder", "POST", {
      initial_dir: input.value.trim() || null,
    });
    if (res.path) {
      input.value = res.path;
    }
  };

  if ($("btn-browse-source")) {
    $("btn-browse-source").onclick = () => browseFolder("source-dir");
  }
  if ($("btn-browse-dest")) {
    $("btn-browse-dest").onclick = () => browseFolder("dest-dir");
  }

  $("btn-start").onclick = async () => {
    const source_dir = $("source-dir").value.trim();
    const dest_dir = $("dest-dir").value.trim();
    if (!source_dir || !dest_dir) {
      return showToast("Indica tu Carpeta con fotos nuevas y Mi Biblioteca Organizada.", "warn");
    }
    const { batch_size, solo_primer_lote } = getBatchOptions();
    const restore = setActionBusy("btn-start", "Organizando...");
    $("progress-stage").textContent = "Estado: Iniciando organización...";
    logActivity(`Iniciando organización (${solo_primer_lote ? "modo prueba" : "completa"})`);
    const timeGapEl = $("time-gap-hours");
    const gpsDistEl = $("gps-distance-km");
    const time_gap_hours = timeGapEl ? Math.max(0.5, Number(timeGapEl.value) || 4) : 4;
    const gps_distance_km = gpsDistEl ? Math.max(1, Number(gpsDistEl.value) || 25) : 25;
    try {
      const snap = await api("/api/scan/start", "POST", {
        source_dir,
        dest_dir,
        batch_size,
        solo_primer_lote,
        time_gap_hours,
        gps_distance_km,
      });
      updateProgressUI(snap);
      connectSSE();
    } catch (err) {
      showToast(`Error al iniciar: ${err.message}`, "warn");
      logActivity(`Error al iniciar: ${err.message}`);
    } finally {
      restore();
    }
  };

  if ($("btn-pause")) {
    $("btn-pause").onclick = async () => updateProgressUI(await api("/api/scan/pause", "POST", {}));
  }
  if ($("btn-resume")) {
    $("btn-resume").onclick = async () => updateProgressUI(await api("/api/scan/resume", "POST", {}));
  }

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

  const setTreeMutationBusy = (folderName, activeBtnId, busyText) => {
    const treeBtnIds = ["btn-rename-folder", "btn-merge-folder", "btn-create-folder"];
    const originals = {};
    for (const id of treeBtnIds) {
      const el = $(id);
      if (!el) continue;
      originals[id] = el.textContent;
      el.disabled = true;
      if (id === activeBtnId && busyText) {
        el.innerHTML = `<span class="spinner" aria-hidden="true"></span> ${busyText}`;
      }
    }
    const nodeEl =
      Array.from(document.querySelectorAll(".folder-item")).find(
        (el) => el.dataset.folder === folderName
      ) || document.querySelector(".folder-item.active");
    if (nodeEl) {
      nodeEl.classList.add("is-loading");
      nodeEl.setAttribute("aria-busy", "true");
    }
    return () => {
      if (nodeEl) {
        nodeEl.classList.remove("is-loading");
        nodeEl.removeAttribute("aria-busy");
      }
      for (const id of treeBtnIds) {
        const el = $(id);
        if (!el) continue;
        el.disabled = false;
        if (originals[id] !== undefined) {
          el.textContent = originals[id];
        }
      }
    };
  };

  $("btn-rename-folder").onclick = async () => {
    if (!state.jobId || !state.selectedFolder) return;
    const next = prompt("Nuevo nombre de carpeta:", state.selectedFolder);
    if (!next) return;
    const restore = setTreeMutationBusy(state.selectedFolder, "btn-rename-folder", "Renombrando...");
    try {
      await api("/api/tree/folders/rename", "POST", {
        job_id: state.jobId,
        old_folder: state.selectedFolder,
        new_folder: next,
      });
      state.selectedFolder = next;
      await loadTree();
    } catch (err) {
      showToast(`Error al renombrar carpeta: ${err.message}`, "warn");
      logActivity(`Error al renombrar carpeta: ${err.message}`);
    } finally {
      restore();
    }
  };

  $("btn-merge-folder").onclick = async () => {
    if (!state.jobId || !state.selectedFolder) return;
    const dest = prompt(`Fusionar '${state.selectedFolder}' dentro de:`, "");
    if (!dest) return;
    const restore = setTreeMutationBusy(state.selectedFolder, "btn-merge-folder", "Fusionando...");
    try {
      await api("/api/tree/folders/merge", "POST", {
        job_id: state.jobId,
        from_folder: state.selectedFolder,
        to_folder: dest,
      });
      state.selectedFolder = dest;
      await loadTree();
    } catch (err) {
      showToast(`Error al fusionar carpeta: ${err.message}`, "warn");
      logActivity(`Error al fusionar carpeta: ${err.message}`);
    } finally {
      restore();
    }
  };

  $("btn-create-folder").onclick = async () => {
    if (!state.jobId) return;
    const folder = prompt("Nombre de la nueva carpeta:", "01_Recuerdos/2024/2024-01 - Nuevo");
    if (!folder) return;
    const restore = setTreeMutationBusy(state.selectedFolder, "btn-create-folder", "Creando...");
    try {
      await api("/api/tree/folders/create", "POST", { job_id: state.jobId, folder });
      await loadTree();
    } catch (err) {
      showToast(`Error al crear carpeta: ${err.message}`, "warn");
      logActivity(`Error al crear carpeta: ${err.message}`);
    } finally {
      restore();
    }
  };

  $("btn-settings").onclick = async () => {
    const cfg = await api("/api/settings");
    $("cfg-exiftool").value = cfg.exiftool_path || "";
    $("cfg-ffmpeg").value = cfg.ffmpeg_path || "";
    $("cfg-antigravity-cli").value = cfg.antigravity_cli || "agy";
    $("cfg-antigravity-model").value = cfg.model || "gemini-4-hydrogen-medium";
    $("settings-modal").showModal();
  };

  $("btn-close-settings").onclick = () => $("settings-modal").close();

  $("settings-form").onsubmit = async (e) => {
    e.preventDefault();
    const res = await api("/api/settings", "POST", {
      exiftool_path: $("cfg-exiftool").value,
      ffmpeg_path: $("cfg-ffmpeg").value,
      antigravity_cli: $("cfg-antigravity-cli").value,
      model: $("cfg-antigravity-model").value,
    });
    $("settings-modal").close();
    if (res && res.warning) {
      showToast(res.warning, "warn");
      logActivity(res.warning);
    }
    await refreshHealth();
  };

  if ($("btn-ai-cancel-run")) {
    $("btn-ai-cancel-run").onclick = async () => {
      try {
        await api("/api/ai/cancel", "POST", {});
        logActivity("Cancelando proceso de IA...");
      } catch (err) {
        showToast(`No se pudo cancelar IA: ${err.message}`, "warn");
      }
    };
  }

  if ($("btn-ai-regenerate")) {
    $("btn-ai-regenerate").onclick = async () => {
      if (!state.jobId) return showToast("Primero ejecuta Organizar Automáticamente.", "warn");
      const restore = setActionBusy("btn-ai-regenerate", "Regenerando nombres con IA...");
      if ($("btn-ai-cancel-run")) $("btn-ai-cancel-run").classList.remove("hidden");
      $("progress-stage").textContent = "Estado: Regenerando nombres con IA...";
      logActivity("Regenerando nombres de carpetas con IA...");
      try {
        const res = await api("/api/ai/regenerate", "POST", { job_id: state.jobId });
        const engine = res.engine || res.model || "Fallback local";
        const msg = res.cancelled
          ? `Regeneración cancelada (${res.updated_folders || 0} carpetas actualizadas).`
          : `Nombres actualizados (${res.updated_folders || 0} carpetas) usando ${engine}.`;
        $("progress-stage").textContent = `Estado: ${msg}`;
        showToast(msg, res.cancelled ? "warn" : "ok");
        logActivity(msg);
        await loadTree();
      } catch (err) {
        showToast(`Error al regenerar nombres: ${err.message}`, "warn");
        logActivity(`Error al regenerar nombres: ${err.message}`);
      } finally {
        if ($("btn-ai-cancel-run")) $("btn-ai-cancel-run").classList.add("hidden");
        restore();
      }
    };
  }

  if ($("btn-ai-classify")) {
    $("btn-ai-classify").onclick = async () => {
      if (!state.jobId || state.items.length === 0) return;
      const ids = state.items.map((it) => it.id);
      const { batch_size, solo_primer_lote } = getBatchOptions();
      const preview = await api("/api/ai/classify", "POST", {
        job_id: state.jobId,
        item_ids: ids,
        confirmed: false,
        batch_size,
        solo_primer_lote,
      });
      $("ai-confirm-text").textContent =
        `Se enviarán ${preview.item_count} elementos en lotes de ${batch_size} al modelo ${preview.model}. ¿Continuar?`;
      $("ai-confirm-modal").showModal();
    };
  }

  $("btn-ai-cancel").onclick = async () => {
    try {
      await api("/api/ai/cancel", "POST", {});
    } catch {
      // ignore if idle
    }
    $("ai-confirm-modal").close();
  };
  $("btn-ai-confirm").onclick = async () => {
    const btn = $("btn-ai-confirm");
    const originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Clasificando...';
    try {
      const ids = state.items.map((it) => it.id);
      const { batch_size, solo_primer_lote } = getBatchOptions();
      const res = await api("/api/ai/classify", "POST", {
        job_id: state.jobId,
        item_ids: ids,
        confirmed: true,
        batch_size,
        solo_primer_lote,
      });
      $("ai-confirm-modal").close();
      if (Array.isArray(res.batches)) {
        renderBatches(res.batches);
      }
      await loadTree();
    } catch (err) {
      showToast(`Error al clasificar con IA: ${err.message}`, "warn");
      logActivity(`Error al clasificar con IA: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalHtml;
    }
  };

  $("btn-export").onclick = async () => {
    if (!state.jobId) return showToast("Primero ejecuta Organizar Automáticamente.", "warn");
    if (!confirm("¿Aplicar y Exportar Árbol al directorio destino?")) return;
    const { batch_size, solo_primer_lote } = getBatchOptions();
    const restore = setActionBusy("btn-export", "Exportando archivos...");
    $("progress-stage").textContent = "Estado: Exportando archivos...";
    logActivity("Iniciando exportación física al directorio destino...");
    try {
      const res = await api("/api/export", "POST", {
        job_id: state.jobId,
        batch_size,
        solo_primer_lote,
      });
      if (Array.isArray(res.batches)) {
        renderBatches(res.batches);
      }
      const msg = `Exportación completa: ${res.exported} exportados, ${res.sidecars_merged || 0} con sidecar .json, ${res.duplicates} duplicados, ${res.skipped} omitidos.`;
      $("progress-stage").textContent = `Estado: ${msg}`;
      showToast(msg, "ok");
      logActivity(msg);
      if ($("btn-open-dest")) {
        $("btn-open-dest").classList.remove("hidden");
      }
      await loadTree();
    } catch (err) {
      showToast(`Error al exportar: ${err.message}`, "warn");
      logActivity(`Error al exportar: ${err.message}`);
    } finally {
      restore();
    }
  };

  if ($("btn-open-dest")) {
    $("btn-open-dest").onclick = async () => {
      try {
        await api("/api/open-destination", "POST", {
          job_id: state.jobId,
          path: $("dest-dir").value.trim() || null,
        });
        logActivity("Carpeta destino abierta en el explorador.");
      } catch (err) {
        showToast(`No se pudo abrir la carpeta: ${err.message}`, "warn");
      }
    };
  }
}

refreshHealth();
bindEvents();
connectSSE();
