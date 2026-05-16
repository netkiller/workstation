import csv
import json
import os
import posixpath
import shutil
import shlex
import subprocess
import threading
import time
import stat as stat_module
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from routes.dataset import normalize_console_log
from routes.project import header_context
from routes.resources import find_resource, read_resources, ssh_connect_kwargs


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
MODEL_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlmodel"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
queue_lock = threading.Lock()
worker_thread = None
running_processes = {}


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


def test_images(project_dir: Path):
    root = project_dir / "test"
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda item: item.relative_to(root).as_posix().lower(),
    )


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
    for candidate in (project_dir / "annotate" / "classes.txt", project_dir / "classes.txt"):
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
        label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
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
    run_name = f"{task_id}-{slug(model['run'] or model['name'])}"
    run_root = project_dir / "test-runs"
    run_dir = run_root / run_name
    command = [
        "yolo",
        "detect",
        "predict",
        f"model={model_path}",
        f"source={project_dir / 'test'}",
        f"project={run_root}",
        f"name={run_name}",
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
        raise RuntimeError(f"{model['name']} 退出码 {return_code}")

    labels_dir = labels_dir_for_run(run_dir)
    detections_by_image = {}
    test_root = project_dir / "test"
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


def sync_tree_delete(sftp, source_root: Path, remote_root: str):
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
            remote_rmtree(sftp, remote_path)
    for source in sorted(source_root.rglob("*"), key=lambda item: item.as_posix().lower()):
        relative = source.relative_to(source_root).as_posix()
        remote_path = posixpath.join(remote_root, relative)
        if source.is_dir():
            sftp_mkdirs(sftp, remote_path)
        elif source.is_file():
            upload_file(sftp, source, remote_path)


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


def download_remote_tree(sftp, remote_path: str, local_path: Path):
    local_path.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote_path):
        if item.filename in {".", ".."}:
            continue
        remote_child = posixpath.join(remote_path, item.filename)
        local_child = local_path / item.filename
        if stat_module.S_ISDIR(item.st_mode):
            download_remote_tree(sftp, remote_child, local_child)
        else:
            local_child.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_child, str(local_child))


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
        remote_test = posixpath.join(remote_project_root, "test")
        remote_base = posixpath.join(remote_project_root, "test-tasks", task_id)
        remote_models = posixpath.join(remote_base, "models")
        remote_runs_root = posixpath.join(remote_base, "test-runs")
        append_log(task_id, f"远程项目目录：{remote_project_root}\n")
        append_log(task_id, f"远程工作目录：{remote_base}\n")
        sftp_mkdirs(sftp, remote_test)
        sftp_mkdirs(sftp, remote_models)
        sftp_mkdirs(sftp, remote_runs_root)

        append_log(task_id, f"同步测试图片：{len(images)} 张（删除远程多余文件）\n")
        sync_tree_delete(sftp, project_dir / "test", remote_test)
        remote_model_paths = {}
        append_log(task_id, f"上传模型：{len(models)} 个\n")
        for index, model in enumerate(models, start=1):
            remote_model = posixpath.join(remote_models, remote_model_filename(index, model))
            upload_file(sftp, Path(model["path"]), remote_model)
            remote_model_paths[model["relative_path"]] = remote_model

        code = remote_exec_stream(client, task_id, "command -v yolo >/dev/null 2>&1")
        if code != 0:
            raise RuntimeError("远程服务器 yolo 命令不存在，请先安装 ultralytics 或确认 PATH。")

        results = []
        for index, model in enumerate(models, start=1):
            run_name = f"{task_id}-{slug(model['run'] or model['name'])}"
            command = [
                "yolo",
                "detect",
                "predict",
                f"model={remote_model_paths[model['relative_path']]}",
                f"source={remote_test}",
                f"project={remote_runs_root}",
                f"name={run_name}",
                "save_txt=True",
                "save_conf=True",
                "exist_ok=True",
                "verbose=False",
            ]
            append_log(task_id, f"远程运行模型 ({index}/{len(models)})：{model['name']}\n")
            code = remote_exec_stream(client, task_id, shell_join(command))
            if code != 0:
                raise RuntimeError(f"{model['name']} 远程退出码 {code}")
            update_task(task_id, progress=round(index / len(models) * 85))

        local_runs_root = project_dir / "test-runs" / task_id
        append_log(task_id, f"下载远程结果：{remote_runs_root}\n")
        download_remote_tree(sftp, remote_runs_root, local_runs_root)
        append_log(task_id, f"已下载到本地：{local_runs_root}\n")

        for model in models:
            run_name = f"{task_id}-{slug(model['run'] or model['name'])}"
            run_dir = local_runs_root / run_name
            labels_dir = labels_dir_for_run(run_dir)
            detections_by_image = {}
            test_root = project_dir / "test"
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
    finally:
        running_processes.pop(task_id, None)
        if sftp is not None:
            sftp.close()
        client.close()


def build_report(task: dict, model_results: list[dict], images: list[Path], project_dir: Path):
    image_names = [image.relative_to(project_dir / "test").as_posix() for image in images]
    rows = []
    for image_name in image_names:
        cells = {}
        for model in model_results:
            cells[model["name"]] = format_detections(model["detections"].get(image_name, []))
        rows.append({"image": image_name, "cells": cells})
    return {
        "task_id": task["id"],
        "project": task["project"],
        "name": task["name"],
        "created_at": task["created_at"],
        "models": [
            {
                "name": item["name"],
                "average_confidence": item["average_confidence"],
                "detection_count": item["detection_count"],
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
    images = test_images(project_dir)
    if not images:
        append_log(task_id, "项目 test 文件夹没有可测试图片。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return
    models = selected_model_items(project_dir, task.get("models", []))
    if not models:
        append_log(task_id, "没有可评估模型。\n")
        update_task(task_id, status="失败", finished_at=datetime.now().isoformat(timespec="seconds"))
        return

    append_log(task_id, f"测试图片：{len(images)} 张\n")
    append_log(task_id, f"评估模型：{len(models)} 个\n\n")
    class_names = read_classes(project_dir)
    results = []
    try:
        if task.get("target_type") == "remote":
            results = run_remote_models(task, project_dir, models, images, class_names)
        else:
            if shutil.which("yolo") is None:
                raise RuntimeError("yolo 命令不存在，请先安装 ultralytics 或确认虚拟环境 PATH。")
            for index, model in enumerate(models, start=1):
                append_log(task_id, f"运行模型 ({index}/{len(models)})：{model['name']}\n")
                results.append(run_model(project_dir, task_id, model, images, class_names))
                update_task(task_id, progress=round(index / len(models) * 90))
    except Exception as error:
        append_log(task_id, f"\n模型测试失败：{error}\n")
        update_task(task_id, status="失败", progress=100, finished_at=datetime.now().isoformat(timespec="seconds"))
        return

    report = build_report(task, results, images, project_dir)
    result_file(task_id).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    append_log(task_id, "\n模型测试评估完成。\n")
    update_task(
        task_id,
        status="完成",
        progress=100,
        result_path=str(result_file(task_id)),
        finished_at=datetime.now().isoformat(timespec="seconds"),
    )


def worker_loop():
    while True:
        with queue_lock:
            task = next((item for item in load_tasks() if item.get("status") == "排队中"), None)
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
    if task.get("status") in {"完成", "失败", "取消"}:
        return 100
    if task.get("status") == "进行中":
        return int(task.get("progress") or 10)
    return 0


def task_view(task: dict):
    view = dict(task)
    view["progress"] = task_progress(view)
    model_count = len(view.get("model_names") or view.get("models") or [])
    if view.get("status") == "完成":
        completed_models = model_count
    else:
        completed_models = min(model_count, int((view["progress"] or 0) / 100 * model_count)) if model_count else 0
    view["completed_models"] = completed_models
    view["model_count"] = model_count
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


@router.get("/test/{project}/reports/{task_id}")
def test_report_page(request: Request, project: str, task_id: str):
    workspace = workspace_path()
    project_dir = project_path(workspace, project)
    report = read_report(task_id)
    task = next((item for item in load_tasks() if item.get("id") == task_id), None)
    if project_dir is None or report is None or not task or task.get("project") != project:
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
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
            "report": report,
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
    models = selected_model_items(project_dir, selected)
    if not models:
        return RedirectResponse(url=f"/test/{project}", status_code=status.HTTP_303_SEE_OTHER)
    if not test_images(project_dir):
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
        "status": "排队中",
        "progress": 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
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
    output = StringIO()
    writer = csv.writer(output)
    model_names = [model["name"] for model in report["models"]]
    writer.writerow(["图片", *model_names])
    for row in report["rows"]:
        writer.writerow([row["image"], *[row["cells"].get(name, "") for name in model_names]])
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
    if process and hasattr(process, "poll") and process.poll() is None:
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
        if task.get("status") not in {"失败", "取消"}:
            return RedirectResponse(url=f"/test/{task.get('project', '')}", status_code=status.HTTP_303_SEE_OTHER)
        task["status"] = "排队中"
        task["progress"] = 0
        task["started_at"] = ""
        task["finished_at"] = ""
        task["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_tasks(tasks)
    result_file(task_id).unlink(missing_ok=True)
    log_file(task_id).write_text(f"任务重新运行: {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    ensure_worker()
    return RedirectResponse(url=f"/test/{task.get('project', '')}", status_code=status.HTTP_303_SEE_OTHER)
