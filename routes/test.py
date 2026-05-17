import csv
import json
import os
import posixpath
import re
import shutil
import shlex
import subprocess
import threading
import time
import stat as stat_module
from datetime import datetime
from io import StringIO
from pathlib import Path
from pathlib import PurePosixPath
from urllib.parse import parse_qs, quote, unquote
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from routes.dataset import normalize_console_log
from routes.dataset import prepare_remote_auth, remote_target, rsync_file_output_args, rsync_ssh_args
from routes.project import (
    append_upload_log,
    build_project_index,
    header_context,
    project_dir,
    relative_log_entry,
    remove_empty_dirs,
    save_upload,
    upload_progress_line,
    upload_relative_path,
    uploaded_files,
)
from routes.resources import find_resource, read_resources, ssh_connect_kwargs
from routes.train import (
    remote_command_output,
    remote_kill_session_command,
    remote_session_backend,
    remote_session_has_command,
    remote_session_start_command,
    read_remote_text,
    tmux_wrap_command,
)


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
MODEL_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlmodel"}
IMAGE_EXTS = {
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
}
queue_lock = threading.Lock()
worker_thread = None
running_processes = {}


TEST_DIR = "test"
TEST_IMAGES_DIR = "images"
TEST_TASKS_DIR = "tasks"
TEST_SETS_META = Path(TEST_DIR) / "sets.json"


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def project_path(workspace: Path, project: str):
    if not project:
        return None
    path = (workspace / project).resolve()
    if path == workspace or not is_inside(path, workspace):
        return None
    return path if path.is_dir() else None


def read_project_name(path: Path):
    meta_file = path / ".project"
    if not meta_file.is_file():
        return path.name
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return str(data.get("name") or path.name)
    except (OSError, json.JSONDecodeError):
        return path.name


def metric_value(row: dict, keywords):
    for key, value in row.items():
        key_text = key.lower().replace(" ", "")
        if all(keyword in key_text for keyword in keywords):
            try:
                return round(float(value), 4)
            except (TypeError, ValueError):
                return None
    return None


def read_metrics(run_dir: Path):
    results_file = run_dir / "results.csv"
    if not results_file.is_file():
        return {}
    try:
        with results_file.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    except OSError:
        return {}
    if not rows:
        return {}
    row = rows[-1]
    return {
        "precision": metric_value(row, ["precision"]),
        "recall": metric_value(row, ["recall"]),
        "map50": metric_value(row, ["map50"]),
        "map5095": metric_value(row, ["map50-95"]),
    }


def run_model_items(project_dir: Path):
    runs_dir = project_dir / "runs"
    if not runs_dir.is_dir():
        return []
    models = []
    for run_dir in sorted((item for item in runs_dir.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        weights_dir = run_dir / "weights"
        model_file = weights_dir / "best.pt"
        if not model_file.is_file():
            model_file = weights_dir / "last.pt"
        if not model_file.is_file() or model_file.suffix.lower() not in MODEL_EXTS:
            continue
        stat = model_file.stat()
        models.append(
            {
                "name": run_dir.name,
                "filename": model_file.name,
                "path": model_file,
                "relative_path": model_file.relative_to(project_dir).as_posix(),
                "run": run_dir.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "metrics": read_metrics(run_dir),
            }
        )
    return sorted(models, key=lambda item: item["path"].stat().st_mtime, reverse=True)


def queue_dir():
    path = workspace_path() / ".test"
    (path / "logs").mkdir(parents=True, exist_ok=True)
    (path / "results").mkdir(parents=True, exist_ok=True)
    return path


def tasks_file():
    return queue_dir() / "tasks.json"


def load_tasks():
    path = tasks_file()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_tasks(tasks: list[dict]):
    tasks_file().write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def update_task(task_id: str, **updates):
    with queue_lock:
        tasks = load_tasks()
        for task in tasks:
            if task.get("id") == task_id:
                task.update(updates)
                break
        save_tasks(tasks)


def log_file(task_id: str):
    return queue_dir() / "logs" / f"{task_id}.log"


def append_log(task_id: str, text: str):
    log_file(task_id).open("a", encoding="utf-8").write(text)


def result_file(task_id: str):
    return queue_dir() / "results" / f"{task_id}.json"


def safe_test_set_name(value: str):
    name = str(value or "").strip()
    if not name or len(name) > 80 or name in {".", ".."}:
        return ""
    if any(char in {"/", "\\"} or ord(char) < 32 for char in name):
        return ""
    return name


def test_sets_meta_path(path: Path):
    return path / TEST_SETS_META


def read_test_sets_meta(path: Path):
    meta_path = test_sets_meta_path(path)
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_test_sets_meta(path: Path, data: dict):
    meta_path = test_sets_meta_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_test_set_meta(path: Path, name: str, description: str = ""):
    meta = read_test_sets_meta(path)
    item = dict(meta.get(name) or {})
    item["name"] = name
    item["description"] = str(description or item.get("description") or "").strip()
    item["updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta[name] = item
    write_test_sets_meta(path, meta)


def rename_test_set(path: Path, old_name: str, new_name: str, description: str = ""):
    old_safe = safe_test_set_name(old_name)
    new_safe = safe_test_set_name(new_name)
    if not old_safe or not new_safe:
        return ""
    root = test_images_dir(path).resolve()
    old_dir = (root / old_safe).resolve()
    new_dir = (root / new_safe).resolve()
    if not is_inside(old_dir, root) or not is_inside(new_dir, root) or not old_dir.is_dir():
        return ""
    if old_safe != new_safe:
        if new_dir.exists():
            return ""
        old_dir.rename(new_dir)
        meta = read_test_sets_meta(path)
        old_item = dict(meta.pop(old_safe, {}) or {})
        write_test_sets_meta(path, meta)
        update_test_set_meta(path, new_safe, description or str(old_item.get("description") or ""))
        with queue_lock:
            tasks = load_tasks()
            changed = False
            for task in tasks:
                if task.get("project") != path.name:
                    continue
                if task.get("test_set") == old_safe:
                    task["test_set"] = new_safe
                    changed = True
                if old_safe in (task.get("test_sets") or []):
                    task["test_sets"] = [new_safe if item == old_safe else item for item in (task.get("test_sets") or [])]
                    changed = True
            if changed:
                save_tasks(tasks)
        return new_safe
    update_test_set_meta(path, new_safe, description)
    return new_safe


def test_images(project_dir: Path, set_names: list[str] | None = None):
    root = test_images_dir(project_dir)
    if not root.is_dir():
        return []
    selected = {str(name) for name in (set_names or []) if str(name)}
    search_roots = [root / name for name in selected if (root / name).is_dir()] if selected else [root]
    return sorted(
        (
            path
            for search_root in search_roots
            for path in search_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda item: item.relative_to(root).as_posix().lower(),
    )


def test_sets(project_dir: Path | None):
    if project_dir is None:
        return []
    root = test_images_dir(project_dir)
    if not root.is_dir():
        return []
    meta = read_test_sets_meta(project_dir)
    colors = ["#2563eb", "#16a34a", "#c47a00", "#0891b2", "#7c3aed", "#0f766e", "#dc2626", "#64748b"]
    sets = []
    for item in sorted((child for child in root.iterdir() if child.is_dir()), key=lambda child: child.name.lower()):
        images = sorted(
            (path for path in item.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
            key=lambda path: path.relative_to(item).as_posix().lower(),
        )
        counts = {}
        for image in images:
            ext = image.suffix.lower().lstrip(".") or "unknown"
            counts[ext] = counts.get(ext, 0) + 1
        extension_items = []
        total = len(images)
        for index, (ext, count) in enumerate(sorted(counts.items(), key=lambda value: (-value[1], value[0]))):
            extension_items.append(
                {
                    "ext": ext,
                    "count": count,
                    "percent": round(count / total * 100) if total else 0,
                    "color": colors[index % len(colors)],
                }
            )
        sets.append(
            {
                "name": item.name,
                "description": str((meta.get(item.name) or {}).get("description") or ""),
                "count": total,
                "extensions": extension_items,
            }
        )
    return sets


def test_set_detail(project_dir: Path | None, name: str):
    safe_name = safe_test_set_name(name)
    if project_dir is None or not safe_name:
        return None
    for item in test_sets(project_dir):
        if item["name"] == safe_name:
            return item
    return None


def test_extension_chart(project_dir: Path | None):
    if project_dir is None:
        return {"total": 0, "style": "conic-gradient(#e2e8f0 0 100%)", "items": []}
    counts = {}
    for image in test_images(project_dir):
        key = image.suffix.lower().lstrip(".") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    colors = ["#1667c7", "#16a34a", "#c47a00", "#0891b2", "#7c3aed", "#0f766e", "#dc2626", "#64748b"]
    cursor = 0.0
    parts = []
    items = []
    for index, (ext, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0]))):
        percent = (count / total * 100) if total else 0
        start = cursor
        end = cursor + percent
        color = colors[index % len(colors)]
        parts.append(f"{color} {start:.2f}% {end:.2f}%")
        items.append({"ext": ext, "count": count, "percent": round(percent), "color": color})
        cursor = end
    return {
        "total": total,
        "style": f"conic-gradient({', '.join(parts)})" if parts else "conic-gradient(#e2e8f0 0 100%)",
        "items": items,
    }


def read_classes(project_dir: Path):
    for candidate in (project_dir / TEST_DIR / "classes.txt", project_dir / "annotate" / "classes.txt", project_dir / "classes.txt"):
        if candidate.is_file():
            items = [line.strip() for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            if items:
                return items
    return []


def slug(value: str):
    text = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return text.strip("._")[:80] or "model"


def selected_model_items(project_dir: Path, selected: list[str]):
    selected_set = set(selected)
    return [model for model in run_model_items(project_dir) if model["relative_path"] in selected_set]


def test_images_dir(project_dir: Path):
    return project_dir / TEST_DIR / TEST_IMAGES_DIR


def test_tasks_dir(project_dir: Path):
    return project_dir / TEST_DIR / TEST_TASKS_DIR


def task_output_name(task: dict):
    return str(task.get("output_name") or slug(str(task.get("name") or task.get("id") or "task")))


def task_output_dir(project_dir: Path, task: dict):
    return test_tasks_dir(project_dir) / task_output_name(task)


def compute_options(workspace: Path):
    return [{"id": "local", "name": "本地", "address": "本机"}] + [
        {"id": item["id"], "name": item["name"], "address": item["address"]}
        for item in read_resources(workspace)
    ]


def parse_label_file(path: Path, class_names: list[str]):
    detections = []
    if not path.is_file():
        return detections
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            confidence = float(parts[-1]) if len(parts) >= 6 else None
        except ValueError:
            continue
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        detections.append({"label": label, "confidence": confidence})
    return detections


def format_detections(detections: list[dict]):
    if not detections:
        return "未识别"
    parts = []
    for item in detections:
        confidence = item.get("confidence")
        suffix = f" {confidence:.3f}" if isinstance(confidence, (int, float)) else ""
        parts.append(f"{item.get('label')}{suffix}")
    return "；".join(parts)


def model_average(detections_by_image: dict[str, list[dict]]):
    values = [
        float(item["confidence"])
        for detections in detections_by_image.values()
        for item in detections
        if isinstance(item.get("confidence"), (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


def labels_dir_for_run(run_dir: Path):
    labels = run_dir / "labels"
    return labels if labels.is_dir() else run_dir


def label_path_for_image(labels_dir: Path, image: Path, test_root: Path):
    relative_label = image.relative_to(test_root).with_suffix(".txt")
    candidates = [
        labels_dir / relative_label,
        labels_dir / f"{image.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(labels_dir.rglob(f"{image.stem}.txt"), key=lambda item: item.as_posix().lower())
    return matches[0] if matches else candidates[0]


def run_model(project_dir: Path, task_id: str, model: dict, images: list[Path], class_names: list[str]):
    model_path = Path(model["path"])
    task = next((item for item in load_tasks() if item.get("id") == task_id), {"id": task_id, "name": task_id})
    selected_set = safe_test_set_name(str(task.get("test_set") or ""))
    source_dir = test_images_dir(project_dir) / selected_set if selected_set else test_images_dir(project_dir)
    run_name = f"{task_id}-{slug(model['run'] or model['name'])}"
    run_root = task_output_dir(project_dir, task)
    run_dir = run_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    command = [
        "yolo",
        "detect",
        "predict",
        f"model={model_path}",
        f"source={source_dir}",
        f"project={run_root}",
        f"name={run_name}",
        "save=True",
        "save_txt=True",
        "save_conf=True",
        "exist_ok=True",
        "verbose=False",
    ]
    append_log(task_id, "$ " + " ".join(str(part) for part in command) + "\n")
    with log_file(task_id).open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=workspace_path(),
            text=True,
        )
        running_processes[task_id] = process
        return_code = process.wait()
    running_processes.pop(task_id, None)
    if return_code != 0:
        log_text = log_file(task_id).read_text(encoding="utf-8", errors="replace") if log_file(task_id).is_file() else ""
        if is_no_space_error(log_text):
            raise RuntimeError(f"{model['name']} 空间不足：No space left on device")
        raise RuntimeError(f"{model['name']} 退出码 {return_code}")

    labels_dir = labels_dir_for_run(run_dir)
    detections_by_image = {}
    test_root = test_images_dir(project_dir)
    for image in images:
        label_path = label_path_for_image(labels_dir, image, test_root)
        detections_by_image[image.relative_to(test_root).as_posix()] = parse_label_file(label_path, class_names)
    return {
        "name": model["name"],
        "run": model.get("run", ""),
        "relative_path": model["relative_path"],
        "detections": detections_by_image,
        "average_confidence": model_average(detections_by_image),
        "detection_count": sum(len(items) for items in detections_by_image.values()),
        "run_dir": str(run_dir),
    }


def shell_join(command: list[str]):
    return " ".join(shlex.quote(str(part)) for part in command)


def sftp_mkdirs(sftp, path: str):
    current = "/" if path.startswith("/") else "."
    for part in [item for item in path.split("/") if item]:
        current = posixpath.join(current, part) if current != "/" else f"/{part}"
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def remote_home_dir(client, sftp):
    try:
        _, stdout, _ = client.exec_command("printf %s \"$HOME\"", timeout=10)
        home = stdout.read().decode("utf-8", errors="replace").strip()
        if home:
            return home
    except Exception:
        pass
    return sftp.normalize(".")


def upload_file(sftp, source: Path, target: str):
    sftp_mkdirs(sftp, posixpath.dirname(target))
    sftp.put(str(source), target)


def transfer_tool(preferred: str = ""):
    rsync = shutil.which("rsync") if preferred in {"", "rsync"} else None
    if rsync and preferred != "scp":
        return "rsync", rsync
    scp = shutil.which("scp") if preferred in {"", "scp"} else None
    if scp:
        return "scp", scp
    return "", ""


def display_command(command: list[str]):
    masked = list(command)
    if masked and Path(masked[0]).name == "sshpass" and len(masked) > 2:
        masked[2] = "******"
    return " ".join(shlex.quote(str(part)) for part in masked)


def run_transfer_command(task_id: str, command: list[str]):
    append_log(task_id, "$ " + display_command(command) + "\n")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_parts = []
    for line in process.stdout:
        output_parts.append(line)
        append_log(task_id, line)
    return process.wait(), "".join(output_parts)


def remote_transfer_base(resource: dict, temp_files: list[Path], task_id: str, preferred: str = ""):
    tool_name, tool_path = transfer_tool(preferred)
    if not tool_name:
        return "", "", [], [], None, "本机未安装 rsync 或 scp"
    prefix, ssh_key, auth_error = prepare_remote_auth(resource, temp_files)
    if auth_error:
        return "", "", [], [], None, auth_error
    append_log(task_id, f"传输工具：{tool_name}\n")
    if tool_name == "rsync":
        base = [*prefix, tool_path, "-auvz", *rsync_file_output_args(tool_path), "-e", rsync_ssh_args(resource, ssh_key)]
    else:
        base = [*prefix, tool_path, "-r", "-P", str(resource.get("port") or 22), "-o", "StrictHostKeyChecking=no"]
        if ssh_key:
            base.extend(["-i", str(ssh_key)])
    return tool_name, tool_path, prefix, base, ssh_key, ""


def rsync_missing_failure(code: int, output: str):
    text = output.lower()
    return code == 127 or "rsync: command not found" in text or "rsync: not found" in text or "rsync: connection unexpectedly closed" in text


def transfer_directory_to_remote(client, resource: dict, task_id: str, source: Path, remote_path: str, delete: bool = False):
    temp_files: list[Path] = []
    try:
        tool_name, _tool_path, _prefix, base, _ssh_key, error = remote_transfer_base(resource, temp_files, task_id)
        if error:
            append_log(task_id, f"{error}，改用 SFTP 传输。\n")
            return False
        target = remote_target(resource, remote_path.rstrip("/") + "/")
        if tool_name == "rsync":
            command = [*base]
            if delete:
                command.append("--delete")
            command.extend([str(source) + "/", target])
        else:
            if delete:
                append_log(task_id, f"scp 不支持 --delete，先清空远程目录：{remote_path}\n")
                cleanup = f"rm -rf {shlex.quote(remote_path)} && mkdir -p {shlex.quote(remote_path)}"
                code, output, error_text = remote_command_output(client, cleanup)
                if output:
                    append_log(task_id, output)
                if error_text:
                    append_log(task_id, error_text)
                if code != 0:
                    raise RuntimeError(f"清空远程目录失败，退出码 {code}")
            command = [*base, str(source / "."), target]
        code, output = run_transfer_command(task_id, command)
        if tool_name == "rsync" and code != 0 and rsync_missing_failure(code, output):
            append_log(task_id, "rsync 不可用，改用 scp 传输。\n")
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            temp_files.clear()
            tool_name, _tool_path, _prefix, base, _ssh_key, error = remote_transfer_base(resource, temp_files, task_id, preferred="scp")
            if error:
                append_log(task_id, f"{error}，改用 SFTP 传输。\n")
                return False
            if delete:
                append_log(task_id, f"scp 不支持 --delete，先清空远程目录：{remote_path}\n")
                cleanup = f"rm -rf {shlex.quote(remote_path)} && mkdir -p {shlex.quote(remote_path)}"
                cleanup_code, cleanup_output, cleanup_error = remote_command_output(client, cleanup)
                if cleanup_output:
                    append_log(task_id, cleanup_output)
                if cleanup_error:
                    append_log(task_id, cleanup_error)
                if cleanup_code != 0:
                    raise RuntimeError(f"清空远程目录失败，退出码 {cleanup_code}")
            code, output = run_transfer_command(task_id, [*base, str(source / "."), target])
        if code != 0:
            raise RuntimeError(f"{tool_name} 传输失败，退出码 {code}")
        return True
    finally:
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def transfer_file_to_remote(client, resource: dict, task_id: str, source: Path, remote_path: str):
    temp_files: list[Path] = []
    try:
        tool_name, _tool_path, _prefix, base, _ssh_key, error = remote_transfer_base(resource, temp_files, task_id)
        if error:
            append_log(task_id, f"{error}，改用 SFTP 传输。\n")
            return False
        code, _output, error_text = remote_command_output(client, f"mkdir -p {shlex.quote(posixpath.dirname(remote_path))}")
        if error_text:
            append_log(task_id, error_text)
        if code != 0:
            raise RuntimeError(f"创建远程目录失败，退出码 {code}")
        target = remote_target(resource, remote_path)
        if tool_name == "rsync":
            command = [*base, str(source), target]
        else:
            command = [*base, str(source), target]
        code, output = run_transfer_command(task_id, command)
        if tool_name == "rsync" and code != 0 and rsync_missing_failure(code, output):
            append_log(task_id, "rsync 不可用，改用 scp 传输。\n")
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            temp_files.clear()
            tool_name, _tool_path, _prefix, base, _ssh_key, error = remote_transfer_base(resource, temp_files, task_id, preferred="scp")
            if error:
                append_log(task_id, f"{error}，改用 SFTP 传输。\n")
                return False
            code, output = run_transfer_command(task_id, [*base, str(source), target])
        if code != 0:
            raise RuntimeError(f"{tool_name} 传输失败，退出码 {code}")
        return True
    finally:
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def transfer_directory_from_remote(resource: dict, task_id: str, remote_path: str, local_path: Path):
    temp_files: list[Path] = []
    try:
        tool_name, _tool_path, _prefix, base, _ssh_key, error = remote_transfer_base(resource, temp_files, task_id)
        if error:
            append_log(task_id, f"{error}，改用 SFTP 下载。\n")
            return False
        local_path.mkdir(parents=True, exist_ok=True)
        if tool_name == "rsync":
            command = [*base, "--delete", remote_target(resource, remote_path.rstrip("/") + "/"), str(local_path) + "/"]
        else:
            append_log(task_id, f"scp 不支持 --delete，先清空本地目录：{local_path}\n")
            if local_path.exists():
                shutil.rmtree(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            command = [*base, remote_target(resource, posixpath.join(remote_path, ".")), str(local_path)]
        code, output = run_transfer_command(task_id, command)
        if tool_name == "rsync" and code != 0 and rsync_missing_failure(code, output):
            append_log(task_id, "rsync 不可用，改用 scp 下载。\n")
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            temp_files.clear()
            tool_name, _tool_path, _prefix, base, _ssh_key, error = remote_transfer_base(resource, temp_files, task_id, preferred="scp")
            if error:
                append_log(task_id, f"{error}，改用 SFTP 下载。\n")
                return False
            append_log(task_id, f"scp 不支持 --delete，先清空本地目录：{local_path}\n")
            if local_path.exists():
                shutil.rmtree(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            code, output = run_transfer_command(task_id, [*base, remote_target(resource, posixpath.join(remote_path, ".")), str(local_path)])
        if code != 0:
            raise RuntimeError(f"{tool_name} 下载失败，退出码 {code}")
        return True
    finally:
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def upload_tree(sftp, source_root: Path, remote_root: str):
    for source in sorted(source_root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root).as_posix()
        upload_file(sftp, source, posixpath.join(remote_root, relative))


def remote_rmtree(sftp, remote_path: str):
    try:
        attrs = sftp.stat(remote_path)
    except OSError:
        return
    if stat_module.S_ISDIR(attrs.st_mode):
        for item in sftp.listdir_attr(remote_path):
            if item.filename in {".", ".."}:
                continue
            remote_rmtree(sftp, posixpath.join(remote_path, item.filename))
        sftp.rmdir(remote_path)
    else:
        sftp.remove(remote_path)


def sync_tree_delete(sftp, source_root: Path, remote_root: str, task_id: str | None = None, on_file=None):
    local_paths = {""}
    for source in source_root.rglob("*"):
        local_paths.add(source.relative_to(source_root).as_posix())
    try:
        remote_items = sorted(
            list_remote_paths(sftp, remote_root),
            key=lambda item: item[0].count("/"),
            reverse=True,
        )
    except OSError:
        remote_items = []
    for relative, remote_path in remote_items:
        if relative not in local_paths:
            if task_id:
                append_log(task_id, f"删除远程多余文件：{relative}\n")
            remote_rmtree(sftp, remote_path)
    sources = sorted(source_root.rglob("*"), key=lambda item: item.as_posix().lower())
    files = [source for source in sources if source.is_file()]
    file_index = 0
    for source in sources:
        relative = source.relative_to(source_root).as_posix()
        remote_path = posixpath.join(remote_root, relative)
        if source.is_dir():
            if task_id:
                append_log(task_id, f"创建远程目录：{relative}/\n")
            sftp_mkdirs(sftp, remote_path)
        elif source.is_file():
            file_index += 1
            if task_id:
                append_log(task_id, f"复制测试文件 ({file_index}/{len(files)})：{relative}\n")
            upload_file(sftp, source, remote_path)
            if on_file:
                on_file(relative)


def list_remote_paths(sftp, remote_root: str):
    items = []
    for item in sftp.listdir_attr(remote_root):
        if item.filename in {".", ".."}:
            continue
        remote_path = posixpath.join(remote_root, item.filename)
        relative = item.filename
        items.append((relative, remote_path))
        if stat_module.S_ISDIR(item.st_mode):
            for child_relative, child_path in list_remote_paths(sftp, remote_path):
                items.append((posixpath.join(relative, child_relative), child_path))
    return items


def count_remote_images(sftp, remote_root: str):
    try:
        return sum(
            1
            for relative, remote_path in list_remote_paths(sftp, remote_root)
            if not stat_module.S_ISDIR(sftp.stat(remote_path).st_mode) and Path(relative).suffix.lower() in IMAGE_EXTS
        )
    except OSError:
        return 0


def remote_file_paths(sftp, remote_root: str):
    try:
        return [
            (relative, remote_path)
            for relative, remote_path in list_remote_paths(sftp, remote_root)
            if not stat_module.S_ISDIR(sftp.stat(remote_path).st_mode)
        ]
    except OSError:
        return []


def download_remote_tree(sftp, remote_path: str, local_path: Path, on_file=None):
    try:
        items = sftp.listdir_attr(remote_path)
    except OSError:
        return
    local_path.mkdir(parents=True, exist_ok=True)
    for item in items:
        if item.filename in {".", ".."}:
            continue
        remote_child = posixpath.join(remote_path, item.filename)
        local_child = local_path / item.filename
        if stat_module.S_ISDIR(item.st_mode):
            download_remote_tree(sftp, remote_child, local_child, on_file=on_file)
        else:
            local_child.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_child, str(local_child))
            if on_file:
                on_file(remote_child, local_child)


def remote_exec_stream(client, task_id: str, command: str):
    append_log(task_id, "$ " + command + "\n")
    _, stdout, stderr = client.exec_command(command)
    channel = stdout.channel
    running_processes[task_id] = channel
    while not channel.exit_status_ready():
        chunks = []
        while channel.recv_ready():
            chunks.append(channel.recv(4096))
        while channel.recv_stderr_ready():
            chunks.append(channel.recv_stderr(4096))
        if chunks:
            append_log(task_id, b"".join(chunks).decode("utf-8", errors="replace"))
        time.sleep(0.5)
    chunks = []
    while channel.recv_ready():
        chunks.append(channel.recv(4096))
    while channel.recv_stderr_ready():
        chunks.append(channel.recv_stderr(4096))
    if chunks:
        append_log(task_id, b"".join(chunks).decode("utf-8", errors="replace"))
    return channel.recv_exit_status()


def remote_model_filename(index: int, model: dict):
    suffix = Path(model["relative_path"]).suffix or ".pt"
    return f"{index:02d}-{slug(model['run'] or model['name'])}{suffix}"


def test_session_name(task_id: str, index: int):
    return f"yoloutils_test_{task_id}_{index:02d}"


def task_attempt(task: dict):
    return max(1, int(task.get("attempt") or 1))


def remote_test_session_name(task: dict, index: int):
    attempt = task_attempt(task)
    suffix = f"_r{attempt:02d}" if attempt > 1 else ""
    return f"{test_session_name(task['id'], index)}{suffix}"


def is_no_space_error(error):
    text = str(error)
    return "[Errno 28]" in text or "No space left on device" in text or "空间不足" in text


def classify_failure_status(error):
    return "空间不足" if is_no_space_error(error) else "失败"


def run_remote_session_command(client, sftp, task_id: str, backend: str, session: str, command: list[str], remote_log: str, remote_exit: str, remote_cwd: str):
    active_code, _, _ = remote_command_output(client, remote_session_has_command(backend, session))
    exit_text = read_remote_text(sftp, remote_exit).strip()
    if active_code != 0 and not exit_text:
        remote_command_output(client, f"mkdir -p {shlex.quote(posixpath.dirname(remote_log))}; : > {shlex.quote(remote_log)}; rm -f {shlex.quote(remote_exit)}")
        wrapped = tmux_wrap_command(shell_join(command), remote_log, remote_exit, remote_cwd)
        start_command = remote_session_start_command(backend, session, wrapped)
        append_log(task_id, "$ " + start_command + "\n")
        code, output, error = remote_command_output(client, start_command)
        if code != 0:
            append_log(task_id, (output + error).strip() + "\n")
            raise RuntimeError(f"{backend} 启动失败，退出码 {code}")
    elif active_code == 0:
        append_log(task_id, f"接回远程会话：{session}\n")

    remote_log_offset = 0
    while True:
        text = read_remote_text(sftp, remote_log)
        if len(text) > remote_log_offset:
            append_log(task_id, text[remote_log_offset:])
            remote_log_offset = len(text)
        exit_text = read_remote_text(sftp, remote_exit).strip()
        if exit_text:
            return int(exit_text) if exit_text.isdigit() else 1
        code, _, _ = remote_command_output(client, remote_session_has_command(backend, session))
        if code != 0:
            text = read_remote_text(sftp, remote_log)
            if len(text) > remote_log_offset:
                append_log(task_id, text[remote_log_offset:])
            exit_text = read_remote_text(sftp, remote_exit).strip()
            return int(exit_text) if exit_text.isdigit() else 1
        time.sleep(1)


def build_results_from_runs(project_dir: Path, task_id: str, models: list[dict], images: list[Path], class_names: list[str]):
    task = next((item for item in load_tasks() if item.get("id") == task_id), {"id": task_id, "name": task_id})
    local_runs_root = task_output_dir(project_dir, task)
    results = []
    for model in models:
        run_name = f"{task_id}-{slug(model['run'] or model['name'])}"
        run_dir = local_runs_root / run_name
        labels_dir = labels_dir_for_run(run_dir)
        detections_by_image = {}
        test_root = test_images_dir(project_dir)
        for image in images:
            label_path = label_path_for_image(labels_dir, image, test_root)
            detections_by_image[image.relative_to(test_root).as_posix()] = parse_label_file(label_path, class_names)
        results.append(
            {
                "name": model["name"],
                "run": model.get("run", ""),
                "relative_path": model["relative_path"],
                "detections": detections_by_image,
                "average_confidence": model_average(detections_by_image),
                "detection_count": sum(len(items) for items in detections_by_image.values()),
                "run_dir": str(run_dir),
            }
        )
    return results


def download_remote_results(task: dict, project_dir: Path, models: list[dict], images: list[Path], class_names: list[str]):
    task_id = task["id"]
    resource = find_resource(workspace_path(), str(task.get("resource_id") or ""))
    if resource is None:
        raise RuntimeError("算力服务器不存在")
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("远程测试需要 paramiko，请先安装 paramiko。") from error

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
    try:
        append_log(task_id, f"连接算力服务器：{resource.get('name') or resource.get('host')}\n")
        client.connect(
            hostname=resource["host"],
            port=int(resource.get("port") or 22),
            username=resource.get("username") or None,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        sftp = client.open_sftp()
        remote_home = remote_home_dir(client, sftp)
        remote_project_root = posixpath.join(remote_home, ".yoloutils", task["project"])
        remote_runs_root = posixpath.join(remote_project_root, TEST_DIR, TEST_TASKS_DIR, task_output_name(task))
        local_runs_root = task_output_dir(project_dir, task)
        append_log(task_id, f"重新下载远程结果：{remote_runs_root}\n")
        run_names = [f"{task_id}-{slug(model['run'] or model['name'])}" for model in models]
        remote_result_groups = [
            (run_name, remote_file_paths(sftp, posixpath.join(remote_runs_root, run_name)))
            for run_name in run_names
        ]
        remote_results = [item for _run_name, items in remote_result_groups for item in items]
        download_total = max(1, len(remote_results))
        download_done = 0
        running_processes[task_id] = client

        def mark_downloaded(remote_path, local_path, count: int = 1):
            nonlocal download_done
            download_done = min(download_total, download_done + max(0, count))
            append_log(task_id, f"下载结果文件 ({download_done}/{download_total})：{remote_path} -> {local_path}\n")
            update_task(
                task_id,
                progress=round(download_done / download_total * 100),
                progress_done=download_done,
                progress_total=download_total,
            )

        update_task(task_id, status="模型下载", progress=0, completed_models=len(models), progress_done=0, progress_total=download_total)
        for run_name, files in remote_result_groups:
            remote_run = posixpath.join(remote_runs_root, run_name)
            local_run = local_runs_root / run_name
            if transfer_directory_from_remote(resource, task_id, remote_run, local_run):
                mark_downloaded(remote_run, local_run, len(files))
            else:
                download_remote_tree(sftp, remote_run, local_run, on_file=mark_downloaded)
        if download_done < download_total:
            update_task(task_id, progress=100, progress_done=download_total, progress_total=download_total)
        append_log(task_id, f"已下载到本地：{local_runs_root}\n")
        return build_results_from_runs(project_dir, task_id, models, images, class_names)
    finally:
        running_processes.pop(task_id, None)
        if sftp is not None:
            sftp.close()
        client.close()


def run_remote_models(task: dict, project_dir: Path, models: list[dict], images: list[Path], class_names: list[str]):
    task_id = task["id"]
    resource = find_resource(workspace_path(), str(task.get("resource_id") or ""))
    if resource is None:
        raise RuntimeError("算力服务器不存在")
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("远程测试需要 paramiko，请先安装 paramiko。") from error

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
    remote_runs_root = ""
    try:
        selected_set = safe_test_set_name(str(task.get("test_set") or ""))
        local_test_images_root = test_images_dir(project_dir)
        local_test_source = local_test_images_root / selected_set if selected_set else local_test_images_root
        if not local_test_source.is_dir():
            raise RuntimeError(f"测试集目录不存在：{local_test_source}")
        deployment_files = [item for item in local_test_source.rglob("*") if item.is_file()]
        deployment_total = max(1, len(deployment_files) + len(models))
        deployment_done = 0

        def mark_deployed(count: int = 1):
            nonlocal deployment_done
            deployment_done += count
            update_task(
                task_id,
                progress=round(deployment_done / deployment_total * 100),
                progress_done=deployment_done,
                progress_total=deployment_total,
            )

        update_task(task_id, status="部署中", progress=0, completed_models=0, progress_done=0, progress_total=deployment_total)
        append_log(task_id, f"连接算力服务器：{resource.get('name') or resource.get('host')}\n")
        client.connect(
            hostname=resource["host"],
            port=int(resource.get("port") or 22),
            username=resource.get("username") or None,
            timeout=10,
            banner_timeout=10,
            auth_timeout=10,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        sftp = client.open_sftp()
        backend = remote_session_backend(client)
        if not backend:
            raise RuntimeError("远程服务器未安装 tmux 或 screen，无法启动持久测试。请先安装其中一个。")
        if backend == "screen":
            append_log(task_id, "远程服务器未安装 tmux，使用 screen 启动持久测试。\n")
        remote_home = remote_home_dir(client, sftp)
        remote_project_root = posixpath.join(remote_home, ".yoloutils", task["project"])
        remote_test = posixpath.join(remote_project_root, TEST_DIR, TEST_IMAGES_DIR)
        remote_test_source = posixpath.join(remote_test, selected_set) if selected_set else remote_test
        remote_base = posixpath.join(remote_project_root, TEST_DIR, TEST_TASKS_DIR, task_output_name(task))
        remote_models = posixpath.join(remote_base, "models")
        remote_runs_root = remote_base
        remote_logs = posixpath.join(remote_base, "logs")
        append_log(task_id, f"远程项目目录：{remote_project_root}\n")
        append_log(task_id, f"远程工作目录：{remote_base}\n")
        sftp_mkdirs(sftp, remote_test)
        sftp_mkdirs(sftp, remote_test_source)
        sftp_mkdirs(sftp, remote_models)
        sftp_mkdirs(sftp, remote_runs_root)
        sftp_mkdirs(sftp, remote_logs)

        source_label = f"test/images/{selected_set}" if selected_set else "test/images"
        append_log(task_id, f"同步测试图片：{source_label}，{len(images)} 张（删除远程多余文件）\n")
        if transfer_directory_to_remote(client, resource, task_id, local_test_source, remote_test_source, delete=True):
            mark_deployed(len(deployment_files))
        else:
            sync_tree_delete(sftp, local_test_source, remote_test_source, task_id, on_file=lambda _relative: mark_deployed())
        remote_image_count = count_remote_images(sftp, remote_test_source)
        append_log(task_id, f"远程测试图片：{remote_image_count} 张\n")
        if remote_image_count <= 0:
            raise RuntimeError(f"远程 test/images 目录没有可测试图片：{remote_test_source}")
        remote_model_paths = {}
        append_log(task_id, f"上传模型：{len(models)} 个\n")
        for index, model in enumerate(models, start=1):
            remote_model = posixpath.join(remote_models, remote_model_filename(index, model))
            append_log(task_id, f"复制模型文件 ({index}/{len(models)})：{model['name']} -> {remote_model}\n")
            if not transfer_file_to_remote(client, resource, task_id, Path(model["path"]), remote_model):
                upload_file(sftp, Path(model["path"]), remote_model)
            mark_deployed()
            remote_model_paths[model["relative_path"]] = remote_model

        code = remote_exec_stream(client, task_id, "command -v yolo >/dev/null 2>&1")
        if code != 0:
            raise RuntimeError("远程服务器 yolo 命令不存在，请先安装 ultralytics 或确认 PATH。")

        update_task(task_id, status="进行中", progress=0, completed_models=0, progress_done=0, progress_total=len(models))
        append_log(task_id, "部署完成，开始远程模型测试。\n")
        results = []
        for index, model in enumerate(models, start=1):
            run_name = f"{task_id}-{slug(model['run'] or model['name'])}"
            command = [
                "yolo",
                "detect",
                "predict",
                f"model={remote_model_paths[model['relative_path']]}",
                f"source={remote_test_source}",
                f"project={remote_runs_root}",
                f"name={run_name}",
                "save=True",
                "save_txt=True",
                "save_conf=True",
                "exist_ok=True",
                "verbose=False",
            ]
            update_task(task_id, completed_models=index)
            append_log(task_id, f"远程运行模型 ({index}/{len(models)})：{model['name']}\n")
            session = remote_test_session_name(task, index)
            attempt_suffix = f"-r{task_attempt(task):02d}" if task_attempt(task) > 1 else ""
            remote_log = posixpath.join(remote_logs, f"{index:02d}-{slug(model['run'] or model['name'])}{attempt_suffix}.log")
            remote_exit = posixpath.join(remote_logs, f"{index:02d}-{slug(model['run'] or model['name'])}{attempt_suffix}.exit")
            running_processes[task_id] = {"type": f"remote-{backend}", "backend": backend, "client": client, "session": session}
            code = run_remote_session_command(client, sftp, task_id, backend, session, command, remote_log, remote_exit, remote_base)
            if code != 0:
                remote_log_text = read_remote_text(sftp, remote_log)
                if is_no_space_error(remote_log_text):
                    raise RuntimeError(f"{model['name']} 远程空间不足：No space left on device")
                raise RuntimeError(f"{model['name']} 远程退出码 {code}")
            update_task(task_id, progress=round(index / len(models) * 100), progress_done=index, progress_total=len(models))

        local_runs_root = task_output_dir(project_dir, task)
        append_log(task_id, f"下载远程结果：{remote_runs_root}\n")
        run_names = [f"{task_id}-{slug(model['run'] or model['name'])}" for model in models]
        remote_result_groups = [
            (run_name, remote_file_paths(sftp, posixpath.join(remote_runs_root, run_name)))
            for run_name in run_names
        ]
        remote_results = [item for _run_name, items in remote_result_groups for item in items]
        download_total = max(1, len(remote_results))
        download_done = 0
        running_processes[task_id] = client

        def mark_downloaded(remote_path, local_path, count: int = 1):
            nonlocal download_done
            download_done = min(download_total, download_done + max(0, count))
            append_log(task_id, f"下载结果文件 ({download_done}/{download_total})：{remote_path} -> {local_path}\n")
            update_task(
                task_id,
                progress=round(download_done / download_total * 100),
                progress_done=download_done,
                progress_total=download_total,
            )

        update_task(task_id, status="模型下载", progress=0, completed_models=len(models), progress_done=0, progress_total=download_total)
        for run_name, files in remote_result_groups:
            remote_run = posixpath.join(remote_runs_root, run_name)
            local_run = local_runs_root / run_name
            if transfer_directory_from_remote(resource, task_id, remote_run, local_run):
                mark_downloaded(remote_run, local_run, len(files))
            else:
                download_remote_tree(sftp, remote_run, local_run, on_file=mark_downloaded)
        if download_done < download_total:
            update_task(task_id, progress=100, progress_done=download_total, progress_total=download_total)
        append_log(task_id, f"已下载到本地：{local_runs_root}\n")

        return build_results_from_runs(project_dir, task_id, models, images, class_names)
    finally:
        running_processes.pop(task_id, None)
        if sftp is not None:
            sftp.close()
        client.close()


def build_report(task: dict, model_results: list[dict], images: list[Path], project_dir: Path):
    image_names = [image.relative_to(test_images_dir(project_dir)).as_posix() for image in images]
    rows = []
    for image_name in image_names:
        cells = {}
        for model in model_results:
            cells[model["name"]] = format_detections(model["detections"].get(image_name, []))
        rows.append({"image": image_name, "cells": cells})
    label_counts = {}
    for model in model_results:
        for detections in model["detections"].values():
            for detection in detections:
                label = str(detection.get("label") or "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
    best_model = max(
        model_results,
        key=lambda item: item["average_confidence"] if item["average_confidence"] is not None else -1,
        default=None,
    )
    return {
        "task_id": task["id"],
        "project": task["project"],
        "name": task["name"],
        "created_at": task["created_at"],
        "image_count": len(image_names),
        "model_count": len(model_results),
        "total_detections": sum(item["detection_count"] for item in model_results),
        "best_model": best_model["name"] if best_model else "",
        "labels": [
            {"label": label, "count": count}
            for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "models": [
            {
                "name": item["name"],
                "run": item.get("run", ""),
                "run_dir": item.get("run_dir", ""),
                "relative_path": item.get("relative_path", ""),
                "average_confidence": item["average_confidence"],
                "detection_count": item["detection_count"],
                "detected_images": sum(1 for detections in item["detections"].values() if detections),
            }
            for item in model_results
        ],
        "rows": rows,
    }


def run_task(task: dict):
    task_id = task["id"]
    update_task(task_id, status="进行中", started_at=datetime.now().isoformat(timespec="seconds"))
    project_dir = project_path(workspace_path(), task["project"])
    if project_dir is None:
        append_log(task_id, "项目不存在。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    selected_set = str(task.get("test_set") or "").strip()
    selected_sets = [selected_set] if selected_set else [str(item) for item in task.get("test_sets") or [] if str(item)]
    images = test_images(project_dir, selected_sets)
    if not images:
        append_log(task_id, "项目 test/images 文件夹没有可测试图片。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    models = selected_model_items(project_dir, task.get("models", []))
    if not models:
        append_log(task_id, "没有可评估模型。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return

    if selected_sets:
        append_log(task_id, f"测试集：{', '.join(selected_sets)}\n")
    append_log(task_id, f"测试图片：{len(images)} 张\n")
    append_log(task_id, f"评估模型：{len(models)} 个\n\n")
    class_names = read_classes(project_dir)
    results = []
    try:
        if task.get("target_type") == "remote" and task.get("download_only"):
            results = download_remote_results(task, project_dir, models, images, class_names)
        elif task.get("target_type") == "remote":
            results = run_remote_models(task, project_dir, models, images, class_names)
        else:
            if shutil.which("yolo") is None:
                raise RuntimeError("yolo 命令不存在，请先安装 ultralytics 或确认虚拟环境 PATH。")
            for index, model in enumerate(models, start=1):
                update_task(task_id, completed_models=index)
                append_log(task_id, f"运行模型 ({index}/{len(models)})：{model['name']}\n")
                results.append(run_model(project_dir, task_id, model, images, class_names))
                update_task(task_id, progress=round(index / len(models) * 100), progress_done=index, progress_total=len(models))
    except Exception as error:
        append_log(task_id, f"\n模型测试失败：{error}\n")
        latest_task = next((item for item in load_tasks() if item.get("id") == task_id), task)
        if latest_task.get("status") == "取消":
            return
        failure_stage = "download" if latest_task.get("status") == "模型下载" else ""
        update_task(
            task_id,
            status=classify_failure_status(error),
            progress=100,
            progress_done=0,
            progress_total=0,
            progress_label="",
            failure_stage=failure_stage,
            download_only=False,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        return

    report = build_report(task, results, images, project_dir)
    result_file(task_id).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(task_id, "\n模型测试评估完成。\n")
    update_task(
        task_id,
        status="完成",
        progress=100,
        progress_done=0,
        progress_total=0,
        progress_label="",
        failure_stage="",
        download_only=False,
        result_path=str(result_file(task_id)),
        finished_at=datetime.now().isoformat(timespec="seconds"),
    )


def worker_loop():
    while True:
        with queue_lock:
            task = next(
                (
                    item
                    for item in load_tasks()
                    if item.get("status") == "排队中"
                    or (item.get("target_type") == "remote" and item.get("status") in {"部署中", "进行中", "模型下载"})
                ),
                None,
            )
        if task is None:
            return
        run_task(task)


def ensure_worker():
    global worker_thread
    if worker_thread and worker_thread.is_alive():
        return
    worker_thread = threading.Thread(target=worker_loop, daemon=True, name="yoloutils-test-worker")
    worker_thread.start()


def task_progress(task: dict):
    if task.get("status") in {"完成", "失败", "取消", "空间不足"}:
        return 100
    if task.get("status") in {"进行中", "部署中", "模型下载"}:
        return int(task.get("progress") if task.get("progress") is not None else 0)
    return 0


def task_view(task: dict):
    view = dict(task)
    view["progress"] = task_progress(view)
    model_count = len(view.get("model_names") or view.get("models") or [])
    if view.get("status") == "完成":
        completed_models = model_count
    elif "completed_models" in view:
        completed_models = min(model_count, max(0, int(view.get("completed_models") or 0))) if model_count else 0
    else:
        completed_models = min(model_count, int((view["progress"] or 0) / 100 * model_count)) if model_count else 0
    view["completed_models"] = completed_models
    view["model_count"] = model_count
    if view.get("status") in {"部署中", "模型下载"}:
        view["progress_done"] = max(0, int(view.get("progress_done") or 0))
        view["progress_total"] = max(0, int(view.get("progress_total") or 0))
    else:
        view["progress_done"] = completed_models
        view["progress_total"] = model_count
    if view.get("progress_total"):
        view["progress_label"] = f"({view['progress_done']}/{view['progress_total']})"
    else:
        view["progress_label"] = ""
    selected_set = str(view.get("test_set") or "").strip()
    if not selected_set:
        selected_set = next((str(item) for item in view.get("test_sets") or [] if str(item)), "")
    if selected_set:
        info = {
            "name": selected_set,
            "description": str(view.get("test_set_description") or ""),
            "count": int(view.get("test_set_image_count") or 0),
        }
        project_dir = project_path(workspace_path(), str(view.get("project") or ""))
        live_info = test_set_detail(project_dir, selected_set) if project_dir else None
        if live_info:
            info["description"] = str(live_info.get("description") or info["description"])
            info["count"] = int(live_info.get("count") or info["count"])
        view["test_set"] = selected_set
        view["test_set_info"] = info
    return view


def project_tasks(project: str):
    with queue_lock:
        tasks = list(reversed(load_tasks()))
    return [task_view(task) for task in tasks if task.get("project") == project]


def read_report(task_id: str):
    path = result_file(task_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def report_source_image_url(project: str, task_id: str, image_index: int):
    return f"/media/thumbnail/reports/{task_id}/{image_index}"


def report_prediction_image_url(project: str, task_id: str, image_index: int, model_index: int):
    return f"/media/thumbnail/reports/{task_id}/{image_index}?model={model_index}"


def report_original_source_image_url(task_id: str, image_index: int):
    return f"/media/original/reports/{task_id}/{image_index}"


def report_original_prediction_image_url(task_id: str, image_index: int, model_index: int):
    return f"/media/original/reports/{task_id}/{image_index}?model={model_index}"


def detection_confidence_values(text: str):
    if not text or text == "未识别":
        return []
    values = []
    for segment in str(text).split("；"):
        parts = segment.strip().split()
        if not parts:
            continue
        try:
            values.append(float(parts[-1]))
        except ValueError:
            continue
    return values


def label_from_token(token: str, class_names: list[str]):
    label = str(token or "").strip()
    if label.startswith("class_") and label[6:].isdigit():
        label = label[6:]
    if label.isdigit():
        index = int(label)
        return class_names[index] if 0 <= index < len(class_names) else label
    return label


def remap_detection_text(text: str, class_names: list[str]):
    if not text or text == "未识别":
        return text or "未识别"
    parts = []
    for segment in str(text).split("；"):
        tokens = segment.strip().split()
        if not tokens:
            continue
        if len(tokens) >= 2:
            label = label_from_token(" ".join(tokens[:-1]), class_names)
            parts.append(f"{label} {tokens[-1]}")
        else:
            parts.append(label_from_token(tokens[0], class_names))
    return "；".join(parts) if parts else "未识别"


def csv_detection_text(text: str, class_names: list[str]):
    if not text or text == "未识别":
        return "-"
    class_index = {name: str(index) for index, name in enumerate(class_names)}
    parts = []
    for segment in str(text).split("；"):
        tokens = segment.strip().split()
        if not tokens:
            continue
        confidence = ""
        label_token = tokens[0]
        if len(tokens) >= 2:
            confidence = tokens[-1]
            label_token = " ".join(tokens[:-1])
        label = str(label_token or "").strip()
        if label.startswith("class_") and label[6:].isdigit():
            label = label[6:]
        elif label in class_index:
            label = class_index[label]
        parts.append(f"{label} {confidence}".strip())
    return "；".join(parts) if parts else "-"


def detection_labels(text: str):
    if not text or text == "未识别":
        return []
    labels = []
    for segment in str(text).split("；"):
        parts = segment.strip().split()
        if not parts:
            continue
        label = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]
        labels.append(label or parts[0])
    return labels


def detection_table_tags(text: str, class_names: list[str]):
    if not text or text == "未识别":
        return []
    tags = []
    class_index = {name: str(index) for index, name in enumerate(class_names)}
    for segment in str(text).split("；"):
        tokens = segment.strip().split()
        if not tokens:
            continue
        confidence = ""
        label_token = tokens[0]
        if len(tokens) >= 2:
            confidence = tokens[-1]
            label_token = " ".join(tokens[:-1])
        raw_label = str(label_token or "").strip()
        normalized = raw_label[6:] if raw_label.startswith("class_") and raw_label[6:].isdigit() else raw_label
        if normalized.isdigit():
            index = normalized
            label = class_names[int(index)] if int(index) < len(class_names) else index
        else:
            index = class_index.get(normalized, normalized)
            label = normalized
        tags.append({"index": index, "label": label, "confidence": confidence})
    return tags


def row_model_score(row: dict, model_name: str):
    values = detection_confidence_values((row.get("cells") or {}).get(model_name, ""))
    return sum(values) / len(values) if values else None


def format_report_rank(item: dict | None):
    if not item:
        return "-"
    return item.get("name") or "-"


def prediction_image_path(run_dir: Path, image_name: str):
    candidates = [
        run_dir / image_name,
        run_dir / Path(image_name).name,
    ]
    image_path = Path(image_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for candidate in sorted(run_dir.rglob(image_path.name), key=lambda item: item.as_posix().lower()):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    return None


def report_model_run_dir(project_dir: Path, report: dict, model: dict):
    run_dir = str(model.get("run_dir") or "").strip()
    if run_dir:
        path = Path(run_dir)
        if path.is_dir():
            return path
    task = next((item for item in load_tasks() if item.get("id") == report.get("task_id")), {})
    output_root = task_output_dir(project_dir, task or {"id": report.get("task_id"), "name": report.get("name")})
    run_name = f"{report.get('task_id')}-{slug(model.get('run') or model.get('name') or '')}"
    path = output_root / run_name
    return path if path.is_dir() else None


def report_view(project: str, project_dir: Path, report: dict):
    view = dict(report)
    rows = []
    class_names = read_classes(project_dir)
    image_count = int(report.get("image_count") or len(report.get("rows") or []))
    total_detections = int(report.get("total_detections") or sum(int(model.get("detection_count") or 0) for model in report.get("models") or []))
    total_slots = image_count * max(1, len(report.get("models") or []))
    raw_rows = report.get("rows") or []
    display_raw_rows = []
    for row in raw_rows:
        row_view = dict(row)
        row_view["cells"] = {
            name: remap_detection_text(value, class_names)
            for name, value in (row.get("cells") or {}).items()
        }
        display_raw_rows.append(row_view)
    labels = list(report.get("labels") or [])
    if not labels:
        label_counts = {}
        for row in display_raw_rows:
            for value in (row.get("cells") or {}).values():
                for label in detection_labels(value):
                    label_counts[label] = label_counts.get(label, 0) + 1
        labels = [
            {"label": label, "count": count}
            for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    else:
        label_counts = {}
        for item in labels:
            label = label_from_token(str(item.get("label") or ""), class_names)
            label_counts[label] = label_counts.get(label, 0) + int(item.get("count") or 0)
        labels = [
            {"label": label, "count": count}
            for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    models = []
    for model in report.get("models") or []:
        item = dict(model)
        model_name = str(item.get("name") or "")
        detected_images = item.get("detected_images")
        if detected_images is None:
            detected_images = sum(
                1
            for row in display_raw_rows
                if (row.get("cells") or {}).get(model_name) and (row.get("cells") or {}).get(model_name) != "未识别"
            )
        detected_images = int(detected_images or 0)
        item["detected_images"] = detected_images
        item["coverage"] = round(detected_images / image_count * 100) if image_count else 0
        item["coverage_style"] = f"conic-gradient(#2563eb 0 {item['coverage']}%, #e2e8f0 {item['coverage']}% 100%)"
        item["confidence_percent"] = round(float(item.get("average_confidence") or 0) * 100)
        item["confidence_label"] = f"{float(item['average_confidence']):.4f}" if item.get("average_confidence") is not None else "-"
        item["run_dir_path"] = report_model_run_dir(project_dir, report, item)
        models.append(item)
    for image_index, row in enumerate(display_raw_rows):
        image_name = str(row.get("image") or "")
        previews = {}
        for model_index, model in enumerate(models):
            run_dir = model.get("run_dir_path")
            predicted = prediction_image_path(run_dir, image_name) if run_dir else None
            if predicted:
                previews[model["name"]] = report_prediction_image_url(project, report["task_id"], image_index, model_index)
        scored_models = [
            {"name": model["name"], "score": row_model_score(row, model["name"])}
            for model in models
        ]
        scored_models = [item for item in scored_models if item["score"] is not None]
        best_model = max(scored_models, key=lambda item: item["score"], default=None)
        worst_model = min(scored_models, key=lambda item: item["score"], default=None)
        row_view = dict(row)
        raw_cells = (raw_rows[image_index].get("cells") or {}) if image_index < len(raw_rows) else {}
        row_view["cell_tags"] = {
            model["name"]: detection_table_tags(raw_cells.get(model["name"], ""), class_names)
            for model in models
        }
        row_view["index"] = image_index
        row_view["image_url"] = report_source_image_url(project, report["task_id"], image_index)
        row_view["original_url"] = report_original_source_image_url(report["task_id"], image_index)
        row_view["previews"] = previews
        row_view["original_previews"] = {
            model["name"]: report_original_prediction_image_url(report["task_id"], image_index, model_index)
            for model_index, model in enumerate(models)
            if model["name"] in previews
        }
        row_view["best_model"] = format_report_rank(best_model)
        row_view["worst_model"] = format_report_rank(worst_model)
        row_view["detected_count"] = sum(1 for value in (row.get("cells") or {}).values() if value and value != "未识别")
        rows.append(row_view)
    view["rows"] = rows
    view["models"] = models
    view["performance_models"] = sorted(
        models,
        key=lambda item: item.get("average_confidence") if item.get("average_confidence") is not None else -1,
        reverse=True,
    )
    longest_model_name = max((len(str(item.get("name") or "")) for item in view["performance_models"]), default=0)
    view["model_name_column_width"] = f"{max(120, min(260, longest_model_name * 9 + 18))}px"
    view["coverage_models"] = sorted(models, key=lambda item: item.get("coverage") or 0, reverse=True)
    view["image_count"] = image_count
    view["model_count"] = int(report.get("model_count") or len(models))
    view["total_detections"] = total_detections
    view["best_model"] = report.get("best_model") or (max(models, key=lambda item: item.get("average_confidence") or -1)["name"] if models else "")
    view["labels"] = labels[:12]
    view["preview_rows"] = rows
    covered_slots = sum(1 for row in rows for value in (row.get("cells") or {}).values() if value and value != "未识别")
    coverage = round(covered_slots / total_slots * 100) if total_slots else 0
    view["coverage"] = {
        "covered": covered_slots,
        "total": total_slots,
        "percent": coverage,
        "style": f"conic-gradient(#2563eb 0 {coverage}%, #e2e8f0 {coverage}% 100%)",
    }
    return view


def report_row_image_name(report: dict, image_index: int):
    rows = report.get("rows") or []
    if image_index < 0 or image_index >= len(rows):
        raise HTTPException(status_code=404, detail="image not found")
    image_name = str(rows[image_index].get("image") or "")
    if not image_name:
        raise HTTPException(status_code=404, detail="image not found")
    return image_name


def report_for_project(project: str, task_id: str):
    workspace = workspace_path()
    project_dir = project_path(workspace, project)
    report = read_report(task_id)
    task = next((item for item in load_tasks() if item.get("id") == task_id), None)
    if project_dir is None or report is None or not task or task.get("project") != project:
        raise HTTPException(status_code=404, detail="report not found")
    return project_dir, report, task


def test_context(request: Request, current_project: str):
    workspace = workspace_path()
    path = project_path(workspace, current_project)
    tasks = project_tasks(current_project) if current_project else []
    return {
        "request": request,
        "workspace": workspace,
        "active_page": "test",
        "current_project": current_project,
        "project_name": read_project_name(path) if path else "",
        "models": run_model_items(path) if path else [],
        "compute_options": compute_options(workspace),
        "image_count": len(test_images(path)) if path else 0,
        "test_sets": test_sets(path),
        "has_test_classes": (path / TEST_DIR / "classes.txt").is_file() if path else False,
        "extension_chart": test_extension_chart(path),
        "tasks": tasks,
        **header_context(request, workspace),
    }


@router.get("/test")
def test_index(request: Request):
    project = request.query_params.get("project", "")
    if project:
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
    current_project = request.cookies.get("current_project", "")
    if not current_project:
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=f"/test/{current_project}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/test/{project}")
def test_with_project(request: Request, project: str):
    response = templates.TemplateResponse(
        request=request,
        name="test/index.html",
        context=test_context(request, project),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


def test_folder_response(request: Request, project: str, set_name: str = ""):
    workspace = workspace_path()
    path = project_path(workspace, project)
    if path is None:
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    safe_name = safe_test_set_name(unquote(set_name)) if set_name else ""
    selected_set = test_set_detail(path, safe_name) if safe_name else None
    if safe_name and selected_set is None:
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
    selected_images = test_images(path, [safe_name]) if safe_name else test_images(path)
    response = templates.TemplateResponse(
        request=request,
        name="test/folder.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "test",
            "current_project": project,
            "project_name": read_project_name(path),
            "folder_mode": "edit" if safe_name else "new",
            "test_set": selected_set or {"name": "", "description": "", "count": 0},
            "image_count": len(selected_images),
            "extension_chart": test_extension_chart(path),
            "has_test_classes": (path / TEST_DIR / "classes.txt").is_file(),
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/test/{project}/folder")
def new_test_folder_page(request: Request, project: str):
    return test_folder_response(request, project)


@router.get("/test/{project}/folder/new")
def legacy_new_test_folder_page(project: str):
    return RedirectResponse(url=f"/test/{project}/folder", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/test/{project}/folder/{set_name}")
def edit_test_folder_page(request: Request, project: str, set_name: str):
    return test_folder_response(request, project, set_name)


@router.post("/project/{directory}/upload/test")
async def upload_test_images(directory: str, request: Request):
    return await upload_test_set_images(directory, "default", request)


@router.post("/project/{directory}/upload/test/{set_name}")
async def upload_test_set_images(directory: str, set_name: str, request: Request):
    if unquote(set_name) == "batch":
        return await upload_test_sets_batch(directory, request)
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    original_set_name = safe_test_set_name(str(request.query_params.get("original") or ""))
    requested_set_name = safe_test_set_name(unquote(set_name))
    safe_set_name = rename_test_set(path, original_set_name, requested_set_name, str(request.query_params.get("description") or "").strip()) if original_set_name else requested_set_name
    if not safe_set_name:
        return JSONResponse({"ok": False, "error": "测试集名称无效"}, status_code=400)
    description = str(request.query_params.get("description") or "").strip()
    flat_only = str(request.query_params.get("flat") or "") == "1"
    update_test_set_meta(path, safe_set_name, description)
    files = await uploaded_files(request)
    target_dir = test_images_dir(path) / safe_set_name

    def stream():
        total = len(files)
        skipped = 0
        saved = []
        error = ""
        yield upload_progress_line(ok=True, stage="saving", saved=0, skipped=0, total=total, progress=0)
        for index, (filename, content) in enumerate(files, start=1):
            relative = upload_relative_path(filename)
            if relative is None or relative.suffix.lower() not in IMAGE_EXTS:
                skipped += 1
            elif flat_only and len(relative.parts) > 1:
                error = "只能上传图片文件，不能携带子目录。"
            else:
                item = save_upload(relative.name, content, target_dir)
                if item is not None:
                    saved.append(item)
            yield upload_progress_line(
                ok=not error,
                stage="saving",
                saved=len(saved),
                skipped=skipped,
                total=total,
                progress=round(index / total * 100) if total else 100,
                file=filename,
                error=error,
            )
            if error:
                return
        append_upload_log(
            path,
            f"上传测试集 {safe_set_name} 图片：接收 {len(files)} 个，保存 {len(saved)} 个，跳过非图片 {skipped} 个",
            [relative_log_entry(path, item) for item in saved],
        )
        index = build_project_index(path)
        yield upload_progress_line(
            ok=True,
            stage="done",
            saved=len(saved),
            skipped=skipped,
            count=int(index.get("test", {}).get("images") or 0),
        )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/project/{directory}/upload/test/batch")
async def upload_test_sets_batch(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    files = await uploaded_files(request)
    root = test_images_dir(path)

    def stream():
        total = len(files)
        skipped = 0
        saved = []
        error = ""
        yield upload_progress_line(ok=True, stage="saving", saved=0, skipped=0, total=total, progress=0)
        for index, (filename, content) in enumerate(files, start=1):
            relative = upload_relative_path(filename)
            if relative is None:
                skipped += 1
            else:
                parts = relative.parts
                if len(parts) < 3:
                    error = "批量上传目录必须包含一级子目录，图片放在一级子目录内。"
                    break
                if len(parts) > 3:
                    error = "一级子目录中不能再包含目录。"
                    break
                set_name = safe_test_set_name(parts[1])
                if not set_name or relative.suffix.lower() not in IMAGE_EXTS:
                    skipped += 1
                else:
                    update_test_set_meta(path, set_name)
                    target = root / set_name / parts[-1]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    saved.append(target)
            yield upload_progress_line(
                ok=not error,
                stage="saving",
                saved=len(saved),
                skipped=skipped,
                total=total,
                progress=round(index / total * 100) if total else 100,
                file=filename,
                error=error,
            )
            if error:
                return
        append_upload_log(
            path,
            f"批量上传测试集：接收 {len(files)} 个，保存 {len(saved)} 个，跳过非图片 {skipped} 个",
            [relative_log_entry(path, item) for item in saved],
        )
        index = build_project_index(path)
        yield upload_progress_line(
            ok=True,
            stage="done",
            saved=len(saved),
            skipped=skipped,
            count=int(index.get("test", {}).get("images") or 0),
        )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.post("/project/{directory}/test-sets")
async def create_test_set(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    body = (await request.body()).decode("utf-8", errors="replace")
    form = parse_qs(body)
    safe_name = safe_test_set_name(str((form.get("name") or [""])[0] or ""))
    description = str((form.get("description") or [""])[0] or "").strip()
    if safe_name:
        target = test_images_dir(path) / safe_name
        target.mkdir(parents=True, exist_ok=True)
        update_test_set_meta(path, safe_name, description)
        append_upload_log(path, f"新建测试集：{safe_name}", [relative_log_entry(path, target)])
        build_project_index(path)
    return RedirectResponse(url=f"/test/{directory}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/project/{directory}/test-sets/{set_name}")
async def update_test_set(directory: str, set_name: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    body = (await request.body()).decode("utf-8", errors="replace")
    form = parse_qs(body)
    new_name = str((form.get("name") or [""])[0] or "")
    description = str((form.get("description") or [""])[0] or "").strip()
    updated_name = rename_test_set(path, unquote(set_name), new_name, description)
    if updated_name:
        append_upload_log(path, f"编辑测试集：{unquote(set_name)} -> {updated_name}", [relative_log_entry(path, test_images_dir(path) / updated_name)])
        build_project_index(path)
    return RedirectResponse(url=f"/test/{directory}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/project/{directory}/test-sets/{set_name}/delete")
async def delete_test_set(directory: str, set_name: str):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    safe_name = safe_test_set_name(unquote(set_name))
    target = (test_images_dir(path) / safe_name).resolve()
    root = test_images_dir(path).resolve()
    if safe_name and target.is_dir() and is_inside(target, root):
        shutil.rmtree(target)
        meta = read_test_sets_meta(path)
        if safe_name in meta:
            meta.pop(safe_name, None)
            write_test_sets_meta(path, meta)
        append_upload_log(path, f"删除测试集：{safe_name}", [relative_log_entry(path, target)])
        build_project_index(path)
    return RedirectResponse(url=f"/test/{directory}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/project/{directory}/upload/test/delete")
async def delete_uploaded_test_images(directory: str):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    test_dir = test_images_dir(path)
    removed = []
    if test_dir.exists():
        for item in test_dir.rglob("*"):
            if not item.is_file():
                continue
            item.unlink()
            removed.append(item)
        remove_empty_dirs(test_dir)
    append_upload_log(
        path,
        f"删除测试图片/文件：{len(removed)} 个",
        [relative_log_entry(path, item) for item in removed],
    )
    index = build_project_index(path)
    return {"ok": True, "deleted": len(removed), "count": int(index.get("test", {}).get("images") or 0)}


@router.post("/project/{directory}/upload/test-classes")
async def upload_test_classes(directory: str, request: Request):
    workspace = workspace_path()
    path = project_dir(workspace, directory)
    if path is None or not path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)

    files = await uploaded_files(request)
    for filename, content in files:
        if PurePosixPath((filename or "").replace("\\", "/")).name.lower() != "classes.txt":
            continue
        target = path / TEST_DIR / "classes.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        append_upload_log(path, "上传测试 classes.txt", [relative_log_entry(path, target)])
        build_project_index(path)
        return {"ok": True, "saved": 1}
    return JSONResponse({"ok": False, "error": "请选择 classes.txt"}, status_code=400)


@router.get("/test/{project}/reports/{task_id}")
def test_report_page(request: Request, project: str, task_id: str):
    workspace = workspace_path()
    try:
        project_dir, report, task = report_for_project(project, task_id)
    except HTTPException:
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
    view_report = report_view(project, project_dir, report)
    response = templates.TemplateResponse(
        request=request,
        name="test/report.html",
        context={
            "request": request,
            "workspace": workspace,
            "active_page": "test",
            "current_project": project,
            "project_name": read_project_name(project_dir),
            "task": task,
            "report": view_report,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.post("/test/run")
async def create_test_task(request: Request):
    form = await request.form()
    project = str(form.get("project") or request.cookies.get("current_project", ""))
    project_dir = project_path(workspace_path(), project)
    if project_dir is None:
        return RedirectResponse(url="/project", status_code=status.HTTP_303_SEE_OTHER)
    selected = [str(value) for key, value in form.multi_items() if key == "models"]
    available_sets = {item["name"]: item for item in test_sets(project_dir)}
    selected_set = safe_test_set_name(str(form.get("test_set") or ""))
    selected_set_info = available_sets.get(selected_set)
    models = selected_model_items(project_dir, selected)
    if not models:
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
    if not selected_set_info or not test_images(project_dir, [selected_set]):
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
    resource_id = str(form.get("resource_id") or "local")
    target_type = "local"
    resource_name = "本地"
    if resource_id != "local":
        resource = find_resource(workspace_path(), resource_id)
        if resource is None:
            return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
        target_type = "remote"
        resource_name = resource.get("name") or resource.get("host") or "算力服务器"
    task = {
        "id": uuid4().hex[:12],
        "name": str(form.get("name") or "").strip() or f"test-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "project": project,
        "target_type": target_type,
        "resource_id": "" if target_type == "local" else resource_id,
        "resource_name": resource_name,
        "models": [model["relative_path"] for model in models],
        "model_names": [model["name"] for model in models],
        "test_set": selected_set,
        "test_sets": [selected_set],
        "test_set_description": str(selected_set_info.get("description") or ""),
        "test_set_image_count": int(selected_set_info.get("count") or 0),
        "status": "排队中",
        "progress": 0,
        "progress_done": 0,
        "progress_total": 0,
        "progress_label": "",
        "completed_models": 0,
        "attempt": 1,
        "failure_stage": "",
        "download_only": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    task["output_name"] = slug(task["name"])
    with queue_lock:
        tasks = load_tasks()
        tasks.append(task)
        save_tasks(tasks)
    append_log(task["id"], f"任务已创建: {task['created_at']}\n")
    ensure_worker()
    return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/test/tasks/{task_id}/logs")
def test_task_logs(task_id: str, offset: int = 0):
    task = next((item for item in load_tasks() if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "任务不存在"}, status_code=404)
    path = log_file(task_id)
    size = path.stat().st_size if path.is_file() else 0
    start = max(0, min(int(offset or 0), size))
    log = ""
    if path.is_file():
        with path.open("rb") as handle:
            handle.seek(start)
            log = handle.read().decode("utf-8", errors="replace")
    view = task_view(task)
    return {"ok": True, "task": view, "log": normalize_console_log(log), "offset": start, "size": size}


@router.get("/test/tasks/{task_id}/csv")
def test_task_csv(task_id: str):
    report = read_report(task_id)
    if report is None:
        return JSONResponse({"ok": False, "error": "报告不存在"}, status_code=404)
    task = next((item for item in load_tasks() if item.get("id") == task_id), {})
    project_dir = project_path(workspace_path(), str(task.get("project") or report.get("project") or ""))
    class_names = read_classes(project_dir) if project_dir is not None else []
    output = StringIO()
    writer = csv.writer(output)
    model_names = [model["name"] for model in report["models"]]
    writer.writerow(["原始图片", *model_names])
    for row in report["rows"]:
        writer.writerow([row["image"], *[csv_detection_text(row["cells"].get(name, ""), class_names) for name in model_names]])
    writer.writerow([])
    writer.writerow(["汇总", *[f"平均置信度 {model['average_confidence'] if model['average_confidence'] is not None else '-'}" for model in report["models"]]])
    writer.writerow(["识别数量", *[model["detection_count"] for model in report["models"]]])
    content = "\ufeff" + output.getvalue()
    filename = f"{report['name']}.csv"
    encoded_filename = quote(filename)
    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"},
    )


@router.post("/test/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    process = running_processes.get(task_id)
    if isinstance(process, dict) and str(process.get("type", "")).startswith("remote-"):
        client = process.get("client")
        backend = str(process.get("backend") or "").strip()
        session = str(process.get("session") or "").strip()
        if client and backend and session:
            try:
                remote_command_output(client, remote_kill_session_command(backend, session))
            except Exception:
                pass
    elif process and hasattr(process, "poll") and process.poll() is None:
        process.terminate()
    elif process and hasattr(process, "close"):
        process.close()
    update_task(task_id, status="取消", progress=100, finished_at=datetime.now().isoformat(timespec="seconds"))
    append_log(task_id, "\n任务已取消。\n")
    task = next((item for item in load_tasks() if item.get("id") == task_id), {})
    return RedirectResponse(url=f"/test/{task.get('project', '')}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/test/tasks/{task_id}/retry")
def retry_task(task_id: str):
    with queue_lock:
        tasks = load_tasks()
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task is None:
            return RedirectResponse(url="/test", status_code=status.HTTP_303_SEE_OTHER)
        if task.get("status") not in {"失败", "取消", "空间不足"}:
            return RedirectResponse(url=f"/test/{task.get('project', '')}", status_code=status.HTTP_303_SEE_OTHER)
        download_only = task.get("status") == "空间不足" and task.get("failure_stage") == "download" and task.get("target_type") == "remote"
        task["status"] = "排队中"
        task["progress"] = 0
        task["progress_done"] = 0
        task["progress_total"] = 0
        task["progress_label"] = ""
        task["completed_models"] = 0
        task["attempt"] = task_attempt(task) + 1
        task["download_only"] = download_only
        task["failure_stage"] = ""
        task["started_at"] = ""
        task["finished_at"] = ""
        task["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_tasks(tasks)
    result_file(task_id).unlink(missing_ok=True)
    log_file(task_id).write_text(f"任务重新运行: {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    ensure_worker()
    return RedirectResponse(url=f"/test/{task.get('project', '')}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/test/tasks/{task_id}/delete")
def delete_task(task_id: str):
    with queue_lock:
        tasks = load_tasks()
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task is None:
            return RedirectResponse(url="/test", status_code=status.HTTP_303_SEE_OTHER)
        project = str(task.get("project") or "")
        if task.get("status") not in {"失败", "取消", "空间不足"}:
            return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
        save_tasks([item for item in tasks if item.get("id") != task_id])

    log_file(task_id).unlink(missing_ok=True)
    result_file(task_id).unlink(missing_ok=True)
    project_dir = project_path(workspace_path(), project)
    if project_dir is not None:
        shutil.rmtree(task_output_dir(project_dir, task), ignore_errors=True)
    return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
