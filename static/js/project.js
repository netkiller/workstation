const createDialog = document.getElementById("createProjectDialog");
const openCreateDialog = document.getElementById("openCreateDialog");
const editDialog = document.getElementById("editProjectDialog");
const editForm = document.getElementById("editProjectForm");
const classesDialog = document.getElementById("classesDialog");
const classesForm = document.getElementById("classesForm");
const editClassesButton = document.getElementById("editClassesButton");
const sftpPanel = document.querySelector("[data-sftp-path]");
const projectIcons = Array.isArray(window.yoloutilsProjectIcons) ? window.yoloutilsProjectIcons : ["▤"];

function randomProjectIcon(input, button, fallback = "") {
  if (!input || !button || !projectIcons.length) {
    return;
  }
  const next = fallback || projectIcons[Math.floor(Math.random() * projectIcons.length)];
  input.value = next;
  button.textContent = next;
}

function copyFeedback(button, text) {
  if (!button || !text) {
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    button.textContent = "✓";
    setTimeout(() => { button.textContent = "⎘"; }, 1200);
  }).catch(() => {
    button.textContent = "!";
    setTimeout(() => { button.textContent = "⎘"; }, 1200);
  });
}

function remoteHost() {
  return window.location.hostname || "127.0.0.1";
}

function remoteUser() {
  return document.querySelector("[data-remote-user]")?.dataset.remoteUser || "";
}

function remoteAuthority() {
  const user = remoteUser();
  return `${user ? `${user}@` : ""}${remoteHost()}`;
}

function remotePath(path) {
  return `${remoteAuthority()}:${path.startsWith("/") ? "" : "/"}${path}`;
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

function renderOnlineUsers(users) {
  const teamList = document.querySelector("[data-team-user-list]");
  const loginList = document.querySelector("[data-login-user-list]");
  const renderDot = (user) => `<span class="online-user-dot" style="--user-color: ${escapeHtml(user.color)}" aria-hidden="true">${escapeHtml(user.initial)}</span>`;
  let onlineCount = users.length;
  if (teamList) {
    const currentProject = teamList.dataset.currentProject || "";
    const visibleUsers = currentProject
      ? users.filter((user) => user.project === currentProject)
      : users;
    onlineCount = visibleUsers.length;
    teamList.innerHTML = visibleUsers.length
      ? visibleUsers.map((user) => `
        <article class="team-user" title="${escapeHtml(user.name)}">
          ${renderDot(user)}
          <div>
            <strong>${escapeHtml(user.name)}</strong>
            <span>${currentProject ? "正在打开项目" : (user.project ? `参与项目：${escapeHtml(user.project_name || user.project)}` : "未打开项目")}</span>
          </div>
        </article>
      `).join("")
      : `<div class="empty compact" data-team-empty>${currentProject ? "暂无在线用户打开该项目" : "暂无在线用户"}</div>`;
  }
  if (loginList) {
    loginList.innerHTML = users.length
      ? users.map((user) => `
        <div class="online-user" title="${escapeHtml(user.name)}">
          ${renderDot(user)}
          <span class="online-user-text">
            <span class="online-user-name">${escapeHtml(user.name)}</span>
            <span class="online-user-project">${user.project ? `打开项目：${escapeHtml(user.project_name || user.project)}` : "未打开项目"}</span>
          </span>
        </div>
      `).join("")
      : '<p data-login-empty>暂无在线用户</p>';
  }
  document.querySelectorAll("[data-online-count]").forEach((item) => {
    item.textContent = `${onlineCount}`;
  });
}

async function heartbeat() {
  if (!document.querySelector(".username-badge")) {
    return;
  }
  try {
    const response = await fetch("/team/heartbeat", {method: "POST"});
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      return;
    }
    renderOnlineUsers(data.users || []);
  } catch (error) {
    // The next heartbeat will retry.
  }
}

heartbeat();
setInterval(heartbeat, 15000);

const teamChat = document.querySelector("[data-team-chat]");
const teamChatMessages = document.querySelector("[data-team-chat-messages]");
const teamChatForm = document.querySelector("[data-team-chat-form]");

function renderTeamChat(messages) {
  if (!teamChat || !teamChatMessages) {
    return;
  }
  const currentUser = teamChat.dataset.currentUser || "";
  teamChatMessages.innerHTML = messages.length
    ? messages.map((message) => `
      <article class="team-chat-message${message.username === currentUser ? " mine" : ""}">
        <span class="online-user-dot" style="--user-color: ${escapeHtml(message.color)}" aria-hidden="true">${escapeHtml(message.initial)}</span>
        <div>
          <div class="team-chat-meta"><strong>${escapeHtml(message.username)}</strong><time>${escapeHtml(message.time)}</time></div>
          <p>${escapeHtml(message.message)}</p>
        </div>
      </article>
    `).join("")
    : '<div class="empty compact" data-team-chat-empty>暂无聊天消息</div>';
  teamChatMessages.scrollTop = teamChatMessages.scrollHeight;
}

async function loadTeamChat() {
  if (!teamChat) {
    return;
  }
  try {
    const response = await fetch("/team/chat", {cache: "no-store"});
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      return;
    }
    renderTeamChat(data.messages || []);
  } catch (error) {
    // The next refresh will retry.
  }
}

if (teamChatForm) {
  teamChatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = teamChatForm.elements.message;
    const message = input.value.trim();
    if (!message) {
      input.focus();
      return;
    }
    input.disabled = true;
    try {
      const response = await fetch("/team/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message}),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) {
        alert(data.error || "发送失败");
        return;
      }
      input.value = "";
      renderTeamChat(data.messages || []);
    } catch (error) {
      alert("发送失败");
    } finally {
      input.disabled = false;
      input.focus();
    }
  });
  teamChatForm.elements.message?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      teamChatForm.requestSubmit();
    }
  });
  loadTeamChat();
  setInterval(loadTeamChat, 3000);
}

if (sftpPanel) {
  const target = sftpPanel.querySelector("[data-sftp-url]");
  const button = sftpPanel.querySelector("[data-copy-sftp]");
  const path = sftpPanel.dataset.sftpPath || "";
  const url = `sftp://${remoteAuthority()}${path.startsWith("/") ? "" : "/"}${path}`;
  if (target) target.textContent = url;
  button?.addEventListener("click", () => {
    copyFeedback(button, url);
  });
}

document.querySelectorAll("[data-rsync-command]").forEach((target) => {
  const container = target.closest("[data-rsync-path]") || target;
  const source = container.dataset.rsyncSource || "./";
  const path = container.dataset.rsyncPath || "";
  const command = `rsync -avz ${source} ${remotePath(path)}`;
  const button = target.closest(".sftp-command-grid, .command-card")?.querySelector("[data-copy-rsync]");
  target.textContent = command;
  button?.addEventListener("click", () => copyFeedback(button, command));
});

if (createDialog && openCreateDialog) {
  const iconInput = createDialog.querySelector("[data-project-icon-value]");
  const iconButton = createDialog.querySelector("[data-random-project-icon]");
  iconButton?.addEventListener("click", () => randomProjectIcon(iconInput, iconButton));
  openCreateDialog.addEventListener("click", () => {
    randomProjectIcon(iconInput, iconButton);
    createDialog.showModal();
  });
  createDialog.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => createDialog.close());
  });
}

if (editDialog && editForm) {
  const editIconInput = editDialog.querySelector("[data-edit-project-icon-value]");
  const editIconButton = editDialog.querySelector("[data-random-edit-project-icon]");
  editIconButton?.addEventListener("click", () => randomProjectIcon(editIconInput, editIconButton));
  editDialog.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => editDialog.close());
  });

  document.querySelectorAll("[data-edit-project]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      editForm.action = `/project/${button.dataset.directory}/edit`;
      editForm.elements.name.value = button.dataset.name || "";
      randomProjectIcon(editIconInput, editIconButton, button.dataset.icon || projectIcons[0] || "▤");
      editForm.elements.description.value = button.dataset.description || "";
      document.querySelectorAll(".menu-panel").forEach((panel) => {
        panel.hidden = true;
      });
      editDialog.showModal();
    });
  });
}

if (classesDialog && classesForm && editClassesButton) {
  editClassesButton.addEventListener("click", () => classesDialog.showModal());
  classesDialog.querySelectorAll("[data-close-classes]").forEach((button) => {
    button.addEventListener("click", () => classesDialog.close());
  });
  classesForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const project = document.querySelector("[data-project]")?.dataset.project;
    const content = classesForm.elements.content.value;
    const response = await fetch(`/project/${project}/classes`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({content}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      alert(data.error || "保存失败");
      return;
    }
    const status = document.querySelector("[data-classes-status]");
    if (status) status.textContent = "已上传";
    const uploadLabel = document.querySelector('[data-upload-kind="classes"] [data-upload-label]');
    if (uploadLabel) uploadLabel.textContent = "已上传";
    editClassesButton.title = "编辑 classes.txt";
    editClassesButton.setAttribute("aria-label", "编辑 classes.txt");
    window.yoloutilsReloadFooterConsole?.();
    classesDialog.close();
  });
}

document.querySelectorAll("[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!confirm(form.dataset.confirm || "确认执行该操作？")) {
      event.preventDefault();
    }
  });
});

document.querySelectorAll(".menu-button").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    const panel = button.parentElement.querySelector(".menu-panel");
    const isHidden = panel.hidden;
    document.querySelectorAll(".menu-panel").forEach((item) => {
      item.hidden = true;
    });
    document.querySelectorAll(".menu-button").forEach((item) => {
      item.setAttribute("aria-expanded", "false");
    });
    panel.hidden = !isHidden;
    button.setAttribute("aria-expanded", String(isHidden));
  });
});

document.addEventListener("click", () => {
  document.querySelectorAll(".menu-panel").forEach((panel) => {
    panel.hidden = true;
  });
  document.querySelectorAll(".menu-button").forEach((button) => {
    button.setAttribute("aria-expanded", "false");
  });
});

function setActionEnabled(selector, enabled) {
  const link = document.querySelector(selector);
  if (link) {
    link.classList.toggle("disabled", !enabled);
  }
}

function setAnnotateReady({imagesReady} = {}) {
  const link = document.querySelector("[data-image-action]");
  if (!link) {
    return;
  }
  if (typeof imagesReady === "boolean") {
    link.dataset.imagesReady = imagesReady ? "1" : "0";
  }
  const ready = link.dataset.imagesReady === "1";
  link.classList.toggle("disabled", !ready);
  link.title = ready ? "进入标注" : "请先上传图片";
}

function dashboardLegendRow(item) {
  return `
    <div class="dashboard-legend-row">
      <span class="legend-swatch" style="background: ${item.color || "#e2e8f0"}" aria-hidden="true"></span>
      <span class="legend-text"><span class="legend-label">${escapeHtml(item.label || "")}</span><span class="legend-metric"><strong>${Number(item.count || 0)}</strong><span>${escapeHtml(item.percent || "0.0%")}</span></span></span>
    </div>
  `;
}

function updateDashboardChart(key, style, items, emptyLabel = "暂无数据") {
  const card = document.querySelector(`[data-dashboard-chart="${key}"]`);
  if (!card) return;
  const pie = card.querySelector(".dashboard-pie");
  if (pie) pie.style.background = style || "#e2e8f0";
  const legend = card.querySelector(".dashboard-legend");
  if (!legend) return;
  const rows = Array.isArray(items) && items.length ? items : [{label: emptyLabel, count: 0, percent: "0.0%", color: "#e2e8f0"}];
  legend.innerHTML = rows.map(dashboardLegendRow).join("");
}

function updateProjectDashboard(dashboard) {
  if (!dashboard) return;
  const progress = Math.max(0, Math.min(100, Number(dashboard.progress_percent || 0)));
  const progressValue = document.querySelector("[data-dashboard-progress-value]");
  if (progressValue) progressValue.textContent = `${Math.round(progress)}%`;
  const progressBar = document.querySelector("[data-dashboard-progress-bar]");
  if (progressBar) progressBar.style.width = `${progress}%`;
  updateDashboardChart("resource", dashboard.resource_chart_style, dashboard.resource_items);
  updateDashboardChart("annotate", dashboard.annotate_chart_style, dashboard.annotate_items);
  updateDashboardChart("imageTypes", dashboard.image_type_chart_style, dashboard.image_type_items, "暂无图片");
}

function fileEntryFile(entry) {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

function readDirectoryEntries(reader) {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

async function filesFromEntry(entry, prefix = "", keepDirectoryName = false) {
  if (entry.isFile) {
    const file = await fileEntryFile(entry);
    return [{file, path: `${prefix}${file.name}`}];
  }

  if (!entry.isDirectory) {
    return [];
  }

  const reader = entry.createReader();
  const files = [];
  const nextPrefix = `${prefix}${keepDirectoryName ? `${entry.name}/` : ""}`;
  while (true) {
    const entries = await readDirectoryEntries(reader);
    if (!entries.length) {
      break;
    }
    for (const child of entries) {
      files.push(...await filesFromEntry(child, nextPrefix, true));
    }
  }
  return files;
}

function filesFromFileList(files) {
  const items = Array.from(files).map((file) => ({
    file,
    path: file.webkitRelativePath || file.name,
  }));
  const relativeItems = items.filter((item) => item.path.includes("/"));
  if (!relativeItems.length || relativeItems.length !== items.length) {
    return items;
  }
  const roots = new Set(relativeItems.map((item) => item.path.split("/")[0]).filter(Boolean));
  if (roots.size !== 1) {
    return items;
  }
  return items.map((item) => ({
    ...item,
    path: item.path.split("/").slice(1).join("/") || item.file.name,
  }));
}

function filesFromFileListKeepRoot(files) {
  return Array.from(files || []).map((file) => ({
    file,
    path: file.webkitRelativePath || file.name,
  }));
}

function validateBatchTestSetFiles(items) {
  const files = Array.from(items || []);
  if (!files.length) return {ok: false, error: "未发现可上传的图片文件。"};
  for (const item of files) {
    const path = String(item.path || item.file?.webkitRelativePath || item.file?.name || "");
    const parts = path.split("/").filter(Boolean);
    if (parts.length < 2) {
      return {ok: false, error: "请选择文件夹批量上传。"};
    }
  }
  return {ok: true};
}

function uploadItemPath(item) {
  const file = item.file || item;
  return String(item.path || file.webkitRelativePath || file.name || "");
}

function uploadItemFile(item) {
  return item.file || item;
}

function uploadFormData(items) {
  const formData = new FormData();
  Array.from(items || []).forEach((item) => {
    const file = uploadItemFile(item);
    formData.append("paths", uploadItemPath(item) || file.name);
    formData.append("files", file, file.name);
  });
  return formData;
}

function compactUploadPath(path, maxLength = 76) {
  const value = String(path || "").replaceAll("\\", "/");
  if (value.length <= maxLength) return value;
  const tail = value.slice(-Math.max(12, maxLength - 3));
  const slashIndex = tail.indexOf("/");
  return `.../${slashIndex >= 0 ? tail.slice(slashIndex + 1) : tail}`;
}

const TEST_BATCH_COLORS = ["#2563eb", "#16a34a", "#c47a00", "#0891b2", "#7c3aed", "#0f766e", "#dc2626", "#64748b"];

function batchUploadSetName(item) {
  const parts = uploadItemPath(item).split("/").filter(Boolean);
  return parts.length >= 3 ? parts[1] : (parts[0] || "default");
}

function initBatchUploadCard(uploads) {
  const card = document.querySelector("[data-test-batch-progress-card]");
  const list = document.querySelector("[data-test-batch-progress-list]");
  const uploadCard = document.querySelector(".test-upload-card");
  if (!card || !list) return null;
  uploadCard?.setAttribute("hidden", "");
  card.hidden = false;
  list.textContent = "";
  const groups = new Map();
  Array.from(uploads || []).forEach((item) => {
    const name = batchUploadSetName(item);
    if (!groups.has(name)) {
      groups.set(name, {name, total: 0, complete: 0, row: null, bar: null, file: null, state: null});
    }
    groups.get(name).total += 1;
  });
  Array.from(groups.values()).forEach((group, index) => {
    const row = document.createElement("article");
    row.className = "test-batch-progress-item";
    row.style.setProperty("--batch-color", TEST_BATCH_COLORS[index % TEST_BATCH_COLORS.length]);
    const content = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `${group.name} 数据集`;
    const progress = document.createElement("em");
    const progressBar = document.createElement("i");
    progress.append(progressBar);
    const file = document.createElement("span");
    file.textContent = "等待上传";
    const state = document.createElement("b");
    state.textContent = "0%";
    content.append(title, progress, file);
    row.append(content, state);
    list.append(row);
    group.row = row;
    group.bar = progressBar;
    group.file = file;
    group.state = state;
  });
  return groups;
}

function restoreBatchUploadCard() {
  document.querySelector("[data-test-batch-progress-card]")?.setAttribute("hidden", "");
  document.querySelector(".test-upload-card")?.removeAttribute("hidden");
}

function updateBatchUploadCard(groups, item) {
  if (!groups) return;
  const group = groups.get(batchUploadSetName(item));
  if (!group) return;
  group.complete += 1;
  const percent = Math.round((group.complete / group.total) * 100);
  if (group.bar) {
    group.bar.style.width = `${percent}%`;
  }
  if (group.file) {
    group.file.textContent = percent >= 100 ? "" : `[${compactUploadPath(uploadItemPath(item), 48)}]`;
  }
  if (group.state) {
    group.state.textContent = percent >= 100 ? "✓" : `${percent}%`;
  }
  group.row?.classList.toggle("complete", percent >= 100);
}

function batchUploadGroups(items) {
  const groups = new Map();
  Array.from(items || []).forEach((item) => {
    const name = batchUploadSetName(item);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(item);
  });
  return groups;
}

function uploadFileName(item) {
  const path = uploadItemPath(item).split("/").filter(Boolean);
  return path[path.length - 1] || uploadItemFile(item).name;
}

function updateUploadConsole(lines, append = "") {
  const consolePanel = window.yoloutilsFooterConsole;
  if (!consolePanel) return;
  if (append && !consolePanel.isOpen?.()) return;
  if (!consolePanel.isOpen?.()) {
    return;
  }
  if (append) {
    consolePanel.append(append);
  } else {
    consolePanel.setText?.(lines.join("\n"));
  }
}

function setUploadPrompt(zone, text) {
  const prompt = zone.querySelector("[data-upload-prompt]");
  if (!prompt) return;
  if (prompt.dataset.uploadDefault == null) {
    prompt.dataset.uploadDefault = prompt.textContent || "";
  }
  prompt.textContent = text || prompt.dataset.uploadDefault;
}

function resetUploadPrompt(zone) {
  const prompt = zone.querySelector("[data-upload-prompt]");
  if (!prompt || prompt.dataset.uploadDefault == null) return;
  prompt.textContent = prompt.dataset.uploadDefault;
}

async function uploadTestFilesByCount(zone, uploadUrl, uploads, batchGroups = null) {
  const total = uploads.length;
  const lines = [`上传测试集：共 ${total} 个文件`, "进度：0%"];
  let latestData = {};
  updateUploadConsole(lines);
  for (const [index, item] of uploads.entries()) {
    const filePath = uploadItemPath(item);
    const displayPath = compactUploadPath(filePath);
    if (!batchGroups) {
      setUploadPrompt(zone, compactUploadPath(filePath, 46));
    }
    const {status, data} = await uploadWithProgress(uploadUrl, uploadFormData([item]), () => {});
    latestData = data || {};
    if (status < 200 || status >= 300 || !latestData.ok) {
      throw new Error(latestData.error || `上传失败：${displayPath}`);
    }
    const complete = index + 1;
    const percent = Math.round((complete / total) * 100);
    setUploadProgress(zone, percent);
    zone.classList.toggle("upload-processing", percent > 0 && percent < 100);
    updateBatchUploadCard(batchGroups, item);
    lines[1] = `进度：${percent}%`;
    updateUploadConsole(lines, `\n[${complete}/${total}] 已上传：${displayPath}`);
  }
  return latestData;
}

async function uploadBatchTestFolders(zone, files) {
  let uploads = Array.from(files || []).filter(isTestImageUpload);
  const validation = validateBatchTestSetFiles(uploads);
  if (!validation.ok) {
    alert(validation.error);
    return;
  }
  const project = document.querySelector("[data-project]")?.dataset.project;
  if (!project) return;

  const batchGroups = initBatchUploadCard(uploads);
  const groupedUploads = batchUploadGroups(uploads);
  const total = uploads.length;
  let completed = 0;
  let latestData = {};
  zone.classList.add("uploading");
  zone.classList.remove("upload-complete", "upload-processing");
  setUploadProgress(zone, 0);
  updateUploadConsole([`批量上传：共 ${groupedUploads.size} 个测试集，${total} 个文件`, "进度：0%"]);

  try {
    await Promise.all(Array.from(groupedUploads.entries()).map(async ([setName, groupItems]) => {
      const uploadUrl = `/project/${encodeURIComponent(project)}/upload/test/${encodeURIComponent(setName)}`;
      for (const item of groupItems) {
        const displayPath = compactUploadPath(uploadItemPath(item));
        const requestItem = {file: uploadItemFile(item), path: uploadFileName(item)};
        const {status, data} = await uploadWithProgress(uploadUrl, uploadFormData([requestItem]), () => {});
        latestData = data || {};
        if (status < 200 || status >= 300 || !latestData.ok) {
          throw new Error(latestData.error || `上传失败：${displayPath}`);
        }
        completed += 1;
        const percent = Math.round((completed / total) * 100);
        setUploadProgress(zone, percent);
        zone.classList.toggle("upload-processing", percent > 0 && percent < 100);
        updateBatchUploadCard(batchGroups, item);
        updateUploadConsole([], `\n[${completed}/${total}] 已上传：${displayPath}`);
      }
    }));
    zone.classList.add("upload-complete");
    zone.classList.remove("upload-processing");
    setUploadProgress(zone, 100);
    const testCounter = document.querySelector("[data-test-count]");
    if (testCounter && typeof latestData.count === "number") {
      testCounter.textContent = `${latestData.count} 个文件`;
    }
    document.querySelector("[data-test-batch-progress-actions]")?.removeAttribute("hidden");
    window.yoloutilsReloadFooterConsole?.();
  } catch (error) {
    restoreBatchUploadCard();
    alert(error.message);
  } finally {
    window.setTimeout(() => {
      zone.classList.remove("uploading", "upload-complete", "upload-processing");
      setUploadProgress(zone, 0);
    }, zone.classList.contains("upload-complete") ? 500 : 0);
  }
}

const TEST_IMAGE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".bmp",
  ".webp",
  ".tif",
  ".tiff",
  ".heic",
  ".heif",
  ".avif",
  ".dng",
  ".mpo",
  ".jp2",
  ".jpeg2000",
]);

function isTestImageUpload(item) {
  const file = item.file || item;
  const path = item.path || file.webkitRelativePath || file.name || "";
  const dotIndex = path.lastIndexOf(".");
  const ext = dotIndex >= 0 ? path.slice(dotIndex).toLowerCase() : "";
  return TEST_IMAGE_EXTENSIONS.has(ext);
}

async function filesFromDataTransfer(dataTransfer, keepRoot = false) {
  const items = Array.from(dataTransfer.items || []);
  const entries = items
    .map((item) => item.webkitGetAsEntry?.())
    .filter(Boolean);

  if (!entries.length) {
    return keepRoot ? filesFromFileListKeepRoot(dataTransfer.files) : filesFromFileList(dataTransfer.files);
  }

  const files = [];
  for (const entry of entries) {
    files.push(...await filesFromEntry(entry, "", keepRoot));
  }
  return files;
}

function validateYoloRunUpload(items) {
  const files = Array.from(items || []);
  if (!files.length) return {ok: false, error: "请选择 YOLO run 目录上传。"};
  const roots = new Set();
  const normalizedPaths = new Set();
  for (const item of files) {
    const path = uploadItemPath(item).replaceAll("\\", "/");
    const parts = path.split("/").filter(Boolean);
    if (parts.length < 2) {
      return {ok: false, error: "请选择完整的 YOLO run 目录，不能只选择单个文件。"};
    }
    roots.add(parts[0]);
    normalizedPaths.add(parts.join("/").toLowerCase());
  }
  if (roots.size !== 1) {
    return {ok: false, error: "一次只能上传一个 YOLO run 目录。"};
  }
  const root = Array.from(roots)[0].toLowerCase();
  const hasBest = normalizedPaths.has(`${root}/weights/best.pt`);
  const hasLast = normalizedPaths.has(`${root}/weights/last.pt`);
  if (!hasBest && !hasLast) {
    return {ok: false, error: "目录结构不符合 YOLO run：必须包含 weights/best.pt 或 weights/last.pt。"};
  }
  return {ok: true};
}

async function uploadFiles(zone, files) {
  const kind = zone.dataset.uploadKind;
  let uploads = Array.from(files);
  if (kind === "model") {
    const validation = validateYoloRunUpload(uploads);
    if (!validation.ok) {
      alert(validation.error);
      return;
    }
  }
  if (kind === "test" || kind?.startsWith("test/")) {
    const before = uploads.length;
    uploads = uploads.filter(isTestImageUpload);
    if (!uploads.length) {
      if (before > 0) alert("未发现可上传的图片文件。");
      return;
    }
  }
  if (zone.dataset.flatImageUpload === "true") {
    const hasDirectory = uploads.some((item) => {
      const file = item.file || item;
      const path = String(item.path || file.webkitRelativePath || file.name || "");
      return path.split("/").filter(Boolean).length > 1;
    });
    if (hasDirectory) {
      alert("只能上传图片文件，不能携带子目录。");
      return;
    }
  }
  if (!uploads.length) {
    return;
  }

  const project = document.querySelector("[data-project]")?.dataset.project;
  if (!project || !kind) {
    return;
  }
  let dynamicUploadUrl = zone.dataset.uploadUrl || "";
  if (zone.dataset.batchUpload === "true") {
    const validation = validateBatchTestSetFiles(uploads);
    if (!validation.ok) {
      alert(validation.error);
      return;
    }
    dynamicUploadUrl = `/project/${project}/upload/test/batch`;
  }
  const uploadUrlTemplate = zone.dataset.uploadUrlTemplate || "";
  if (!dynamicUploadUrl && uploadUrlTemplate) {
    const nameInput = document.querySelector(zone.dataset.testSetNameInput || "");
    const descriptionInput = document.querySelector(zone.dataset.testSetDescriptionInput || "");
    const setName = (nameInput?.value || "").trim();
    if (!setName) {
      alert("请先填写测试集名称。");
      nameInput?.focus();
      return;
    }
    if (setName.length > 80 || setName === "." || setName === ".." || /[\\/]|[\x00-\x1f]/.test(setName)) {
      alert("测试集名称不能包含路径分隔符。");
      nameInput?.focus();
      return;
    }
    dynamicUploadUrl = uploadUrlTemplate.replace("__SET_NAME__", encodeURIComponent(setName));
    const description = (descriptionInput?.value || "").trim();
    if (description) {
      dynamicUploadUrl += `${dynamicUploadUrl.includes("?") ? "&" : "?"}description=${encodeURIComponent(description)}`;
    }
  }
  if (dynamicUploadUrl && !uploadUrlTemplate) {
    const descriptionInput = document.querySelector(zone.dataset.testSetDescriptionInput || "");
    const description = (descriptionInput?.value || "").trim();
    if (description && !dynamicUploadUrl.includes("description=")) {
      dynamicUploadUrl += `${dynamicUploadUrl.includes("?") ? "&" : "?"}description=${encodeURIComponent(description)}`;
    }
  }

  const formData = uploadFormData(uploads);
  const batchGroups = zone.dataset.batchUpload === "true" ? initBatchUploadCard(uploads) : null;
  zone.classList.add("uploading");
  zone.classList.remove("upload-complete", "upload-processing");
  setUploadProgress(zone, 0);

  try {
    const uploadUrl = dynamicUploadUrl || `/project/${project}/upload/${kind}`;
    let data;
    if (kind === "images" || kind === "test" || kind?.startsWith("test/")) {
      data = await uploadTestFilesByCount(zone, uploadUrl, uploads, batchGroups);
    } else {
      const result = await uploadWithProgress(uploadUrl, formData, (percent) => {
        setUploadProgress(zone, percent);
        zone.classList.toggle("upload-processing", percent > 0 && percent < 100);
      });
      data = result.data;
      if (result.status < 200 || result.status >= 300 || !data.ok) {
        throw new Error(data.error || "上传失败");
      }
    }
    zone.classList.add("upload-complete");
    zone.classList.remove("upload-processing");
    setUploadProgress(zone, 100);
    if (kind === "images") {
      document.querySelector("[data-image-count]").textContent = `${data.count} 个文件`;
      setAnnotateReady({imagesReady: data.count > 0});
      updateProjectDashboard(data.dashboard);
    } else if (kind === "test" || kind?.startsWith("test/")) {
      const testCounter = document.querySelector("[data-test-count]");
      if (testCounter) testCounter.textContent = `${data.count} 个文件`;
      if (data.saved === 0 && data.skipped > 0) {
        alert("未保存新图片，已跳过非图片文件。");
      }
      if (document.querySelector("[data-test-page]")) {
        const redirectUrl = zone.dataset.redirectAfterUpload;
        if (zone.dataset.batchUpload !== "true") {
          window.setTimeout(() => {
            if (redirectUrl) window.location.href = redirectUrl;
            else window.location.reload();
          }, 500);
        }
      }
    } else if (kind === "model") {
      document.querySelector("[data-model-count]").textContent = `${data.count} 个`;
      setActionEnabled("[data-model-action]", data.count > 0);
      window.setTimeout(() => window.location.reload(), 500);
    } else if (kind === "classes" || kind === "test-classes") {
      const label = zone.querySelector("[data-upload-label]");
      if (label) {
        label.textContent = kind === "classes" ? "已上传" : "classes.txt 已上传";
      }
      const status = document.querySelector(kind === "test-classes" ? "[data-test-classes-status]" : "[data-classes-status]");
      if (status) status.textContent = "已上传";
      if (kind === "classes") {
        const editButton = document.querySelector("#editClassesButton");
        if (editButton) {
          editButton.title = "编辑 classes.txt";
          editButton.setAttribute("aria-label", "编辑 classes.txt");
        }
      }
      if (kind === "test-classes" && document.querySelector("[data-test-page]")) {
        window.setTimeout(() => window.location.reload(), 500);
      }
    }
    window.yoloutilsReloadFooterConsole?.();
  } catch (error) {
    if (batchGroups) restoreBatchUploadCard();
    alert(error.message);
  } finally {
    window.setTimeout(() => {
      zone.classList.remove("uploading", "upload-complete", "upload-processing");
      setUploadProgress(zone, 0);
      resetUploadPrompt(zone);
    }, zone.classList.contains("upload-complete") ? 500 : 0);
  }
}

async function clearUploadedImages(button) {
  const project = document.querySelector("[data-project]")?.dataset.project;
  if (!project) {
    return;
  }
  const zone = button.closest("[data-upload-zone]");
  const kind = zone?.dataset.uploadKind || button.dataset.clearUpload || "images";
  const isTestUpload = kind === "test";
  const message = isTestUpload ? "确认删除已上传测试图片？" : "确认删除已上传图片和对应标注文件？classes.txt 会保留。";
  if (!window.confirm(message)) {
    return;
  }

  button.disabled = true;
  try {
    const endpoint = isTestUpload ? "test" : "images";
    const response = await fetch(`/project/${encodeURIComponent(project)}/upload/${endpoint}/delete`, {
      method: "POST",
      headers: {"Accept": "application/json"},
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "删除失败");
    }
    if (isTestUpload) {
      const counter = document.querySelector("[data-test-count]");
      if (counter) {
        counter.textContent = `${data.count} 个文件`;
      }
      if (document.querySelector("[data-test-page]")) {
        window.setTimeout(() => window.location.reload(), 300);
      }
    } else {
      document.querySelector("[data-image-count]").textContent = `${data.count} 个文件`;
      setAnnotateReady({imagesReady: data.count > 0});
      updateProjectDashboard(data.dashboard);
    }
    zone?.classList.remove("uploading", "upload-complete", "dragging");
    if (zone) setUploadProgress(zone, 0);
    window.yoloutilsReloadFooterConsole?.();
  } catch (error) {
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

function setUploadProgress(zone, percent) {
  const value = Math.max(0, Math.min(100, Math.round(percent)));
  zone.querySelector(".upload-icon")?.style.setProperty("--upload-progress", `${value}%`);
  const symbol = zone.querySelector(".upload-symbol");
  if (!symbol) return;
  if (symbol.dataset.uploadDefaultHtml == null) {
    symbol.dataset.uploadDefaultHtml = symbol.innerHTML || "";
  }
  const showPercent = zone.classList.contains("uploading") && !zone.classList.contains("upload-complete");
  if (showPercent) {
    symbol.textContent = `${value}%`;
  } else {
    symbol.innerHTML = symbol.dataset.uploadDefaultHtml;
  }
  symbol.classList.toggle("upload-symbol-percent", showPercent);
}

function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    let parsedLength = 0;
    let responseBuffer = "";
    let latestData = {};
    const usesServerProgress = url.includes("/upload/images") || url.includes("/upload/test");
    request.open("POST", url);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      if (usesServerProgress) {
        if (event.loaded > 0 && event.loaded < event.total) onProgress(4);
        return;
      }
      onProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.addEventListener("progress", () => {
      const contentType = request.getResponseHeader("content-type") || "";
      if (!contentType.includes("application/x-ndjson")) return;
      responseBuffer += request.responseText.slice(parsedLength);
      parsedLength = request.responseText.length;
      const lines = responseBuffer.split("\n");
      responseBuffer = lines.pop() || "";
      lines.filter(Boolean).forEach((line) => {
        try {
          const data = JSON.parse(line);
          latestData = data;
          if (typeof data.progress === "number") onProgress(data.progress);
        } catch (error) {
          // Ignore partial NDJSON chunks; the next progress event will complete them.
        }
      });
    });
    request.addEventListener("load", () => {
      let data = {};
      try {
        const contentType = request.getResponseHeader("content-type") || "";
        if (contentType.includes("application/x-ndjson")) {
          request.responseText.split("\n").filter(Boolean).forEach((line) => {
            latestData = JSON.parse(line);
          });
          data = latestData;
        } else {
          data = JSON.parse(request.responseText || "{}");
        }
      } catch (error) {
        const contentType = request.getResponseHeader("content-type") || "未知类型";
        const responseText = String(request.responseText || "").trim();
        const summary = responseText
          ? responseText.replace(/\s+/g, " ").slice(0, 240)
          : "服务器返回空响应";
        reject(new Error(`上传响应解析失败：HTTP ${request.status || 0}，${contentType}，${summary}`));
        return;
      }
      resolve({status: request.status, data});
    });
    request.addEventListener("error", () => reject(new Error("上传失败")));
    request.addEventListener("abort", () => reject(new Error("上传已取消")));
    request.send(formData);
  });
}

document.querySelectorAll("[data-upload-zone]").forEach((zone) => {
  const input = zone.querySelector("[data-file-input]") || zone.querySelector("input");
  const directoryInput = zone.querySelector("[data-directory-input]");
  const batchDirectoryInput = zone.querySelector("[data-batch-directory-input]");
  const directoryButton = zone.querySelector("[data-directory-button]");
  const batchDirectoryButton = zone.querySelector("[data-batch-directory-button]");
  const clearButton = zone.querySelector("[data-clear-upload]");
  const browseButton = zone.querySelector("[data-browse-media]");

  zone.addEventListener("click", (event) => {
    if (zone.classList.contains("uploading")) {
      event.preventDefault();
      return;
    }
    if (event.target.closest("input")) {
      return;
    }
    if (event.target.closest("[data-directory-button]")) {
      return;
    }
    if (event.target.closest("[data-batch-directory-button]")) {
      return;
    }
    if (event.target.closest("[data-clear-upload]")) {
      return;
    }
    if (event.target.closest("[data-browse-media]")) {
      return;
    }
    if (event.target.closest("#editClassesButton")) {
      return;
    }
    input.click();
  });
  zone.addEventListener("keydown", (event) => {
    if (zone.classList.contains("uploading")) {
      event.preventDefault();
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    input.click();
  });
  if (input !== directoryInput) {
    input.addEventListener("change", () => uploadFiles(zone, filesFromFileList(input.files)));
    input.addEventListener("click", () => {
      input.value = "";
    });
  }
  directoryInput?.addEventListener("change", () => {
    const fileItems = zone.dataset.uploadKind === "model"
      ? filesFromFileListKeepRoot(directoryInput.files)
      : filesFromFileList(directoryInput.files);
    uploadFiles(zone, fileItems);
  });
  directoryInput?.addEventListener("click", () => {
    directoryInput.value = "";
  });
  batchDirectoryInput?.addEventListener("change", () => {
    uploadBatchTestFolders(zone, filesFromFileListKeepRoot(batchDirectoryInput.files));
  });
  batchDirectoryInput?.addEventListener("click", () => {
    batchDirectoryInput.value = "";
  });
  directoryButton?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (zone.classList.contains("uploading")) return;
    directoryInput?.click();
  });
  batchDirectoryButton?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (zone.classList.contains("uploading")) return;
    batchDirectoryInput?.click();
  });
  clearButton?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    clearUploadedImages(clearButton);
  });
  browseButton?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    window.location.href = browseButton.href;
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (zone.classList.contains("uploading")) return;
    zone.classList.add("dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", async (event) => {
    event.preventDefault();
    if (zone.classList.contains("uploading")) return;
    zone.classList.remove("dragging");
    uploadFiles(zone, await filesFromDataTransfer(event.dataTransfer, zone.dataset.uploadKind === "model"));
  });
});
