import os
import errno
import json
import pty
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from routes.project import header_context
from routes.resources import find_resource, read_resources


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
JPEG_EXTS = {".jpg", ".jpeg"}
DATASET_NAME_PATTERN = re.compile(r"^[^\x00-\x1f\x7f/\\:]+$")
ANNOTATE_DIR = "annotate"
DEPLOY_MODES = {"full": "全量", "incremental": "增量", "sync": "同步", "diff": "同步"}
DEPLOY_TARGETS = {"local": "本地", "remote": "远程"}
DEFAULT_DATASET_ICON = "▦"
ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_UNICODE_SYMBOLS = tuple(
    symbol
    for codepoint in range(0x110000)
    for symbol in (chr(codepoint),)
    if unicodedata.category(symbol).startswith("S")
)
DATASET_ICONS = (DEFAULT_DATASET_ICON,) + tuple(symbol for symbol in _UNICODE_SYMBOLS if symbol != DEFAULT_DATASET_ICON)
deploy_lock = threading.Lock()
build_lock = threading.Lock()
dataset_worker_lock = threading.Lock()
dataset_worker_thread: threading.Thread | None = None
running_build_tasks: set[str] = set()
running_deploy_tasks: set[str] = set()
BUILD_RUNNABLE_STATUSES = {"排队中", "创建中"}
DEPLOY_RUNNABLE_STATUSES = {"排队中", "进行中"}
ACTIVE_DEPLOY_STATUSES = {"排队中", "进行中", "等待确认"}
LOG_TIMESTAMP_PREFIX = re.compile(r"^\[[^\]]+\]\s*")


def workspace_path():
    workspace = os.environ.get("YOLOUTILS_WORKSPACE")
    return Path(workspace).expanduser().resolve() if workspace else Path.cwd().resolve()


def count_split(split_dir: Path):
    if not split_dir.is_dir():
        return {"images": 0, "labels": 0}
    images = sum(1 for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    labels = sum(1 for path in split_dir.rglob("*.txt") if path.is_file())
    return {"images": images, "labels": labels}


def count_dataset_split(dataset_path: Path, split: str):
    images_dir = dataset_path / "images" / split
    labels_dir = dataset_path / "labels" / split
    if images_dir.is_dir() or labels_dir.is_dir():
        images = sum(1 for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS) if images_dir.is_dir() else 0
        labels = sum(1 for path in labels_dir.rglob("*.txt") if path.is_file()) if labels_dir.is_dir() else 0
        return {"images": images, "labels": labels}
    return count_split(dataset_path / split)


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def project_dir(workspace: Path, directory: str):
    path = (workspace / directory).resolve()
    if path == workspace or not is_inside(path, workspace):
        return None
    return path


def project_name(project_dir: Path):
    meta = project_dir / ".project"
    if not meta.is_file():
        return project_dir.name
    try:
        import json

        data = json.loads(meta.read_text(encoding="utf-8"))
        return str(data.get("name") or project_dir.name)
    except (OSError, ValueError):
        return project_dir.name


def current_project_from_request(request: Request, fallback: str = ""):
    return fallback or request.cookies.get("current_project", "")


def image_files(root: Path):
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def split_counts(files: list[Path]):
    return {
        "images": len(files),
        "labels": sum(1 for path in files if path.with_suffix(".txt").is_file()),
    }


def dataset_chart_data(splits: dict):
    total_images = sum(split["images"] for split in splits.values())
    total_labels = sum(split["labels"] for split in splits.values())
    if total_images:
        train_percent = splits["train"]["images"] / total_images * 100
        val_percent = splits["val"]["images"] / total_images * 100
        test_percent = splits["test"]["images"] / total_images * 100
    else:
        train_percent = val_percent = test_percent = 0
    train_end = train_percent
    val_end = train_percent + val_percent
    chart_style = (
        f"conic-gradient(#1667c7 0 {train_end:.2f}%, "
        f"#16a34a {train_end:.2f}% {val_end:.2f}%, "
        f"#f59e0b {val_end:.2f}% 100%)"
        if total_images
        else "conic-gradient(#e2e8f0 0 100%)"
    )
    return {
        "total_images": total_images,
        "total_labels": total_labels,
        "chart_style": chart_style,
        "chart_segments": [
            {"name": "train", "count": splits["train"]["images"], "percent": round(train_percent)},
            {"name": "val", "count": splits["val"]["images"], "percent": round(val_percent)},
            {"name": "test", "count": splits["test"]["images"], "percent": round(test_percent)},
        ],
    }


def expected_dataset_preview(project_path: Path, task: dict):
    files = image_files(project_path / ANNOTATE_DIR)
    total = len(files)
    try:
        val_percent = int(task.get("val_percent", 20) or 20)
        test_percent = int(task.get("test_percent", 0) or 0)
    except (TypeError, ValueError):
        val_percent = 20
        test_percent = 0
    test_count = round(total * test_percent / 100)
    val_count = round(total * val_percent / 100)
    test_files = files[:test_count]
    val_files = files[test_count : test_count + val_count]
    train_files = files[test_count + val_count :]
    splits = {
        "train": split_counts(train_files),
        "val": split_counts(val_files),
        "test": split_counts(test_files),
    }
    return {"splits": splits, **dataset_chart_data(splits)}


def copy_image_with_label(source: Path, source_root: Path, target_root: Path):
    relative = source.relative_to(source_root)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    label = source.with_suffix(".txt")
    if label.is_file():
        shutil.copy2(label, target.with_suffix(".txt"))


def copy_image_to_dataset(source: Path, source_root: Path, image_root: Path, label_root: Path):
    relative = source.relative_to(source_root)
    image_target = image_root / relative
    label_target = label_root / relative.with_suffix(".txt")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image_target)
    label = source.with_suffix(".txt")
    if label.is_file():
        shutil.copy2(label, label_target)


def project_classes_file(project_path: Path):
    path = project_path / ANNOTATE_DIR / "classes.txt"
    return path if path.is_file() else None


def project_classes_preview(project_path: Path | None):
    if project_path is None or not project_path.is_dir():
        return {"path": f"{ANNOTATE_DIR}/classes.txt", "exists": False, "items": []}
    path = project_path / ANNOTATE_DIR / "classes.txt"
    items = []
    if path.is_file():
        items = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
    return {"path": f"{ANNOTATE_DIR}/classes.txt", "exists": path.is_file(), "items": items}


def dataset_icon(value: str):
    return value.strip() or DEFAULT_DATASET_ICON


def dataset_meta_path(dataset_path: Path):
    return dataset_path / ".dataset.json"


def read_dataset_meta(dataset_path: Path):
    path = dataset_meta_path(dataset_path)
    if not path.is_file():
        return {"icon": DEFAULT_DATASET_ICON, "created_at": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"icon": DEFAULT_DATASET_ICON, "created_at": ""}
    return {
        "icon": dataset_icon(str(data.get("icon") or "")),
        "created_at": str(data.get("created_at") or ""),
    }


def write_dataset_meta(dataset_path: Path, icon: str, created_at: str = ""):
    created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_meta_path(dataset_path).write_text(
        json.dumps({"icon": dataset_icon(icon), "created_at": created_at}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def datetime_sort_value(value: str, fallback: float = 0):
    value = (value or "").strip()
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value.replace(" ", "T")).timestamp()
    except ValueError:
        return fallback


def build_tasks_file(project_path: Path):
    return project_path / ".dataset-builds.json"


def build_logs_root(project_path: Path):
    path = project_path / ".dataset-build-logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_build_tasks(project_path: Path):
    path = build_tasks_file(project_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    return [task for task in tasks if isinstance(task, dict)]


def write_build_tasks(project_path: Path, tasks: list[dict]):
    build_tasks_file(project_path).write_text(
        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_build_task(project_path: Path, task_id: str, **updates):
    with build_lock:
        tasks = read_build_tasks(project_path)
        for task in tasks:
            if task.get("id") == task_id:
                task.update(updates)
                task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        write_build_tasks(project_path, tasks)


def active_build_tasks(project_path: Path):
    tasks = []
    for task in read_build_tasks(project_path):
        status_value = task.get("status")
        dataset_path = project_path / "datasets" / str(task.get("name", ""))
        if status_value == "完成" and dataset_path.is_dir():
            continue
        if status_value not in {"排队中", "创建中"} and not dataset_path.is_dir():
            continue
        tasks.append(task)
    return tasks


def task_key(project_path: Path, task: dict):
    return f"{project_path.resolve()}::{task.get('id', '')}"


def workspace_project_dirs(workspace: Path):
    if not workspace.is_dir():
        return []
    return [path for path in sorted(workspace.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]


def claim_next_dataset_task(workspace: Path):
    with dataset_worker_lock:
        for current_project_path in workspace_project_dirs(workspace):
            for task in read_build_tasks(current_project_path):
                if task.get("status") not in BUILD_RUNNABLE_STATUSES:
                    continue
                key = task_key(current_project_path, task)
                if key in running_build_tasks:
                    continue
                running_build_tasks.add(key)
                return "build", current_project_path, dict(task), key
            for task in read_deploy_tasks(current_project_path):
                if task.get("status") not in DEPLOY_RUNNABLE_STATUSES:
                    continue
                key = task_key(current_project_path, task)
                if key in running_deploy_tasks:
                    continue
                running_deploy_tasks.add(key)
                return "deploy", current_project_path, dict(task), key
    return None


def dataset_worker_loop(workspace_value: str):
    global dataset_worker_thread
    workspace = Path(workspace_value)
    idle_checks = 0
    try:
        while True:
            claimed = claim_next_dataset_task(workspace)
            if claimed is None:
                idle_checks += 1
                if idle_checks >= 4:
                    return
                time.sleep(0.5)
                continue
            idle_checks = 0
            kind, current_project_path, task, key = claimed
            try:
                if kind == "build":
                    run_build_dataset_task(current_project_path, task)
                else:
                    run_deploy_task(current_project_path, task)
            finally:
                with dataset_worker_lock:
                    if kind == "build":
                        running_build_tasks.discard(key)
                    else:
                        running_deploy_tasks.discard(key)
    finally:
        with dataset_worker_lock:
            dataset_worker_thread = None


def ensure_dataset_worker(workspace: Path | None = None):
    global dataset_worker_thread
    workspace = workspace or workspace_path()
    with dataset_worker_lock:
        if dataset_worker_thread and dataset_worker_thread.is_alive():
            return
        dataset_worker_thread = threading.Thread(
            target=dataset_worker_loop,
            args=(str(workspace),),
            daemon=True,
            name="dataset-background-worker",
        )
        dataset_worker_thread.start()


def latest_dataset_deploy_task(project_path: Path, dataset_name: str):
    tasks = [
        task
        for task in read_deploy_tasks(project_path)
        if str(task.get("dataset") or "") == dataset_name
    ]
    return tasks[0] if tasks else None


def dataset_deploy_defaults(project_path: Path, dataset_name: str):
    latest = latest_dataset_deploy_task(project_path, dataset_name)
    if latest:
        return {
            "resource_id": str(latest.get("resource_id") or ""),
            "mode": "sync",
            "target_path": str(latest.get("target_path") or f"~/datasets/{dataset_name}"),
        }
    resources = read_resources(workspace_path())
    if not resources:
        return None
    return {
        "resource_id": str(resources[0].get("id") or ""),
        "mode": "sync",
        "target_path": f"~/datasets/{dataset_name}",
    }


def deploy_mode_icon(mode: str):
    return "↻" if mode in {"sync", "diff", "incremental"} else "⬢"


def build_dataset(workspace: Path, project: str, name: str, val_percent: int, test_percent: int, icon: str = ""):
    name = (name or "").strip()
    if not name or name in {".", ".."} or not DATASET_NAME_PATTERN.match(name):
        return None, "数据集名称不能包含路径分隔符、冒号或控制字符"
    if val_percent < 0 or test_percent < 0 or val_percent + test_percent > 100:
        return None, "val 和 test 百分比之和不能超过 100"

    project_path = project_dir(workspace, project)
    if project_path is None or not project_path.is_dir():
        return None, "项目不存在"

    images_root = project_path / ANNOTATE_DIR
    dataset_dir = project_path / "datasets" / name
    if dataset_dir.exists():
        return None, "数据集已存在"
    classes_file = project_classes_file(project_path)
    if classes_file is None:
        return None, f"缺少 {ANNOTATE_DIR}/classes.txt，不能创建数据集"

    files = image_files(images_root)
    total = len(files)
    test_count = round(total * test_percent / 100)
    val_count = round(total * val_percent / 100)
    test_files = files[:test_count]
    val_files = files[test_count : test_count + val_count]
    train_files = files[test_count + val_count :]

    for split, split_files in (("train", train_files), ("val", val_files), ("test", test_files)):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for source in split_files:
            copy_image_to_dataset(source, images_root, images_dir, labels_dir)

    shutil.copy2(classes_file, dataset_dir / "classes.txt")
    write_dataset_meta(dataset_dir, icon)

    return {
        "path": str(dataset_dir),
        "total": total,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }, None


def run_build_dataset_task(project_path: Path, task: dict):
    workspace = workspace_path()
    task_id = task["id"]
    log_path = Path(task.get("log_path") or build_logs_root(project_path) / f"{task_id}.log")
    previous_status = str(task.get("status") or "")
    try:
        update_build_task(project_path, task_id, status="创建中", progress=3, error="")
        append_deploy_log(log_path, f"开始创建数据集 {task['name']}")
        name = task["name"]
        val_percent = int(task.get("val_percent", 20) or 20)
        test_percent = int(task.get("test_percent", 0) or 0)
        icon = task.get("icon", "")
        source_root = project_path / ANNOTATE_DIR
        dataset_path = project_path / "datasets" / name
        files = image_files(source_root)
        total = len(files)
        append_deploy_log(log_path, f"待复制图片：{total} 个")
        if dataset_path.exists() and previous_status != "创建中":
            raise RuntimeError("数据集已存在")
        if val_percent < 0 or test_percent < 0 or val_percent + test_percent > 100:
            raise RuntimeError("val 和 test 百分比之和不能超过 100")
        classes_file = project_classes_file(project_path)
        if classes_file is None:
            raise RuntimeError(f"缺少 {ANNOTATE_DIR}/classes.txt，不能创建数据集")
        test_count = round(total * test_percent / 100)
        val_count = round(total * val_percent / 100)
        split_groups = (
            ("test", files[:test_count]),
            ("val", files[test_count : test_count + val_count]),
            ("train", files[test_count + val_count :]),
        )
        for split, split_files in split_groups:
            append_deploy_log(log_path, f"{split}：{len(split_files)} 个")
        copied = 0
        for split, split_files in split_groups:
            images_dir = dataset_path / "images" / split
            labels_dir = dataset_path / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for source in split_files:
                copy_image_to_dataset(source, source_root, images_dir, labels_dir)
                copied += 1
                append_deploy_log(log_path, f"{split}/{source.relative_to(source_root)}")
                progress = 5 + round((copied / total) * 90) if total else 95
                update_build_task(project_path, task_id, progress=min(progress, 95))
        shutil.copy2(classes_file, dataset_path / "classes.txt")
        append_deploy_log(log_path, "classes.txt")
        write_dataset_meta(dataset_path, icon, str(task.get("created_at") or ""))
        update_build_task(project_path, task_id, status="完成", progress=100, error="")
        append_deploy_log(log_path, "数据集创建完成")
        if task.get("deploy_enabled"):
            deploy_form = {
                "resource_id": str(task.get("resource_id") or ""),
                "mode": str(task.get("deploy_mode") or "sync"),
                "target_path": str(task.get("target_path") or f"~/datasets/{name}"),
            }
            deploy_task, deploy_error = create_deploy_task(project_path, dataset_path, name, deploy_form)
            if deploy_error:
                update_build_task(project_path, task_id, deploy_error=deploy_error)
                append_deploy_log(log_path, f"创建部署任务失败：{deploy_error}")
            elif deploy_task:
                update_build_task(project_path, task_id, deploy_task_id=deploy_task["id"])
                append_deploy_log(log_path, f"已创建部署任务：{deploy_task['id']}")
    except Exception as error:
        message = build_error_message(error)
        if message == "空间不足" and dataset_path.exists():
            shutil.rmtree(dataset_path, ignore_errors=True)
            append_deploy_log(log_path, "空间不足，已停止复制并清理未完成数据集。")
        update_build_task(project_path, task_id, status="失败", error=message)
        append_deploy_log(log_path, f"数据集创建失败：{message}")


def create_build_task(
    project_path: Path,
    name: str,
    val_percent: int,
    test_percent: int,
    icon: str,
    deploy_enabled: bool = False,
    deploy_mode: str = "sync",
    resource_id: str = "",
):
    name = (name or "").strip()
    if not name or name in {".", ".."} or not DATASET_NAME_PATTERN.match(name):
        return None, "数据集名称不能包含路径分隔符、冒号或控制字符"
    if val_percent < 0 or test_percent < 0 or val_percent + test_percent > 100:
        return None, "val 和 test 百分比之和不能超过 100"
    dataset_path = project_path / "datasets" / name
    if dataset_path.exists():
        return None, "数据集已存在"
    if any(task.get("name") == name and task.get("status") in {"排队中", "创建中"} for task in read_build_tasks(project_path)):
        return None, "数据集正在创建"
    if project_classes_file(project_path) is None:
        return None, f"缺少 {ANNOTATE_DIR}/classes.txt，不能创建数据集"
    if deploy_enabled:
        if deploy_mode not in {"full", "incremental", "sync"}:
            return None, "部署方式不正确"
        if find_resource(workspace_path(), resource_id) is None:
            return None, "请选择算力服务器"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    log_path = build_logs_root(project_path) / f"{task_id}.log"
    task = {
        "id": task_id,
        "name": name,
        "icon": dataset_icon(icon),
        "val_percent": val_percent,
        "test_percent": test_percent,
        "status": "排队中",
        "progress": 0,
        "error": "",
        "deploy_enabled": bool(deploy_enabled),
        "deploy_mode": deploy_mode if deploy_mode in {"full", "incremental", "sync"} else "sync",
        "resource_id": resource_id if deploy_enabled else "",
        "target_path": f"~/datasets/{name}" if deploy_enabled else "",
        "log_path": str(log_path),
        "created_at": now,
        "updated_at": now,
    }
    with build_lock:
        tasks = read_build_tasks(project_path)
        tasks.insert(0, task)
        write_build_tasks(project_path, tasks)
    append_deploy_log(log_path, "数据集创建任务已创建")
    ensure_dataset_worker(workspace_path())
    return task, ""


def dataset_dir(workspace: Path, project: str, name: str):
    project_path = project_dir(workspace, project)
    if project_path is None or not project_path.is_dir():
        return None
    datasets_root = (project_path / "datasets").resolve()
    path = (datasets_root / name).resolve()
    if path == datasets_root or not is_inside(path, datasets_root) or not path.is_dir():
        return None
    return path


def deploy_root(project_path: Path):
    path = project_path / ".dataset-deploy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def deploy_tasks_file(project_path: Path):
    return deploy_root(project_path) / "tasks.json"


def read_deploy_tasks(project_path: Path):
    path = deploy_tasks_file(project_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    return [task for task in tasks if isinstance(task, dict)]


def write_deploy_tasks(project_path: Path, tasks: list[dict]):
    deploy_tasks_file(project_path).write_text(
        json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_deploy_task(project_path: Path, task_id: str, **updates):
    with deploy_lock:
        tasks = read_deploy_tasks(project_path)
        for task in tasks:
            if task.get("id") == task_id:
                task.update(updates)
                task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        write_deploy_tasks(project_path, tasks)


def append_deploy_log(log_path: Path, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as output:
        output.write(f"[{timestamp}] {message}\n")


def normalize_console_log(text: str):
    text = ANSI_ESCAPE_PATTERN.sub("", text or "")
    text = text.replace("\r\n", "\n")
    cleaned_lines = []
    for line in text.split("\n"):
        if "\r" in line:
            segments = [segment for segment in line.split("\r") if segment.strip()]
            line = segments[-1] if segments else ""
        cleaned_lines.append(CONTROL_CHAR_PATTERN.sub("", line))
    return "\n".join(cleaned_lines)


def current_copy_file_from_log(log_text: str):
    ignored_prefixes = (
        "$ ",
        "rsync ",
        "openrsync",
        "receiving incremental file list",
        "sent ",
        "total size ",
        "部署任务已创建",
        "开始部署",
        "待复制文件",
        "执行 ",
        "部署完成",
        "部署失败",
        "失败任务重试",
        "已确认删除覆盖",
        "未知部署方式",
        "数据集创建任务已创建",
        "开始创建",
        "待复制图片",
        "数据集创建完成",
        "数据集创建失败",
        "train：",
        "val：",
        "test：",
        "空间不足",
        "创建部署任务失败",
        "已创建部署任务",
        "重新创建数据集",
    )
    for line in reversed(log_text.splitlines()):
        clean = LOG_TIMESTAMP_PREFIX.sub("", line).strip()
        if not clean or clean.endswith("/") or clean.startswith(ignored_prefixes):
            continue
        return clean
    return ""


def deploy_current_file_label(log_text: str, status_value: str):
    current_file = current_copy_file_from_log(log_text)
    if current_file:
        return current_file
    if status_value in ACTIVE_DEPLOY_STATUSES:
        return "等待复制文件..."
    if status_value == "完成":
        return "没有文件需要复制"
    return ""


def current_copy_file_for_task(task: dict):
    log_path = Path(str(task.get("log_path") or ""))
    if not log_path.is_file():
        return ""
    return current_copy_file_from_log(log_path.read_text(encoding="utf-8", errors="replace"))


def deploy_current_file_for_task(task: dict):
    log_path = Path(str(task.get("log_path") or ""))
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return deploy_current_file_label(log_text, str(task.get("status") or ""))


def extract_transfer_percent(text: str):
    matches = re.findall(r"(?<!\d)(\d{1,3})%", text)
    if not matches:
        return None
    value = max(0, min(int(matches[-1]), 100))
    return value


def count_deploy_files(source: Path):
    if not source.is_dir():
        return 0
    return sum(1 for path in source.rglob("*") if path.is_file())


def is_no_space_error(error: BaseException):
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return True
        current = current.__cause__ or current.__context__
    return "No space left on device" in str(error)


def build_error_message(error: BaseException):
    return "空间不足" if is_no_space_error(error) else str(error)


def normalize_task_error(error: str):
    text = str(error or "")
    return "空间不足" if "No space left on device" in text else text


def dataset_select_items(workspace: Path, project: str):
    return [{"name": item["name"], "path": item["path"]} for item in dataset_items(workspace, project)]


def deploy_target_label(task: dict):
    if task.get("target_type") == "remote":
        return task.get("resource_name") or "远程服务器"
    return "本地"


def deploy_task_view(task: dict):
    mode = str(task.get("mode") or "")
    source_path = Path(str(task.get("source_path") or ""))
    icon = task.get("icon") or (read_dataset_meta(source_path)["icon"] if source_path.is_dir() else DEFAULT_DATASET_ICON)
    status_value = task.get("status", "")
    return {
        **task,
        "icon": icon,
        "mode_label": DEPLOY_MODES.get(mode, mode),
        "mode_icon": deploy_mode_icon(mode),
        "target_label": deploy_target_label(task),
        "target_type_label": DEPLOY_TARGETS.get(task.get("target_type"), task.get("target_type", "")),
        "completed": status_value == "完成",
        "needs_overwrite_confirm": status_value == "等待确认" and task.get("mode") == "full",
        "can_retry": status_value == "失败",
        "current_file": deploy_current_file_for_task(task) if status_value in ACTIVE_DEPLOY_STATUSES else "",
    }


def remote_target(resource: dict, target_path: str):
    return f"{resource.get('username')}@{resource.get('host')}:{target_path}"


def remote_shell_path(target_path: str):
    if target_path == "~":
        return "$HOME"
    if target_path.startswith("~/"):
        return "$HOME/" + shlex.quote(target_path[2:])
    return shlex.quote(target_path)


def rsync_ssh_args(resource: dict, key_file: Path | None = None):
    args = ["ssh", "-p", str(resource.get("port") or 22), "-o", "StrictHostKeyChecking=no"]
    if key_file:
        args.extend(["-i", str(key_file)])
    return " ".join(args)


def prepare_remote_auth(resource: dict, temp_files: list[Path]):
    key_file = None
    if resource.get("use_private_key") and resource.get("private_key"):
        temp = tempfile.NamedTemporaryFile(prefix="dataset-deploy-key-", delete=False)
        key_file = Path(temp.name)
        temp.write(resource["private_key"].encode("utf-8"))
        temp.close()
        key_file.chmod(0o600)
        temp_files.append(key_file)
        return [], key_file, ""
    password = resource.get("password") or ""
    if password:
        sshpass = shutil.which("sshpass")
        if not sshpass:
            return [], None, "远程服务器使用密码认证，但本机未安装 sshpass，无法执行远程部署。请改用私钥或安装 sshpass。"
        return [sshpass, "-p", password], None, ""
    return [], None, ""


def run_command(
    command: list[str],
    log_path: Path,
    progress_callback=None,
    progress_total: int = 0,
    count_output_files: bool = False,
    use_pty: bool = False,
):
    display = list(command)
    if display and Path(display[0]).name == "sshpass" and len(display) > 2:
        display[2] = "******"
    append_deploy_log(log_path, "$ " + " ".join(display))
    master_fd = None
    if use_pty:
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        os.close(slave_fd)
        stdout_fd = master_fd
    else:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        stdout_fd = process.stdout.fileno()
    os.set_blocking(stdout_fd, False)
    pending = ""
    last_logged_progress = None
    last_callback_progress = None
    copied_files = 0

    def handle_output_segment(segment: str, force: bool = False):
        nonlocal last_logged_progress, last_callback_progress, copied_files
        clean = segment.strip()
        if not clean:
            return
        if count_output_files:
            if clean.endswith("/"):
                return
            copied_files += 1
            if progress_callback and progress_total > 0:
                percent = max(0, min(round(copied_files / progress_total * 100), 100))
                if percent != last_callback_progress:
                    progress_callback(percent)
                    last_callback_progress = percent
            append_deploy_log(log_path, clean)
            return
        percent = extract_transfer_percent(clean)
        if percent is not None:
            if progress_callback and percent != last_callback_progress:
                progress_callback(percent)
                last_callback_progress = percent
            should_log = force or percent == 100 or last_logged_progress is None or abs(percent - last_logged_progress) >= 5
            if should_log:
                append_deploy_log(log_path, clean)
                last_logged_progress = percent
            return
        append_deploy_log(log_path, clean)

    def consume_available_output():
        nonlocal pending
        consumed = False
        while True:
            try:
                chunk = os.read(stdout_fd, 4096)
            except BlockingIOError:
                break
            except OSError as error:
                if error.errno == errno.EIO and use_pty:
                    break
                raise
            if not chunk:
                break
            consumed = True
            pending += chunk.decode("utf-8", errors="replace")
            parts = re.split(r"[\r\n]+", pending)
            pending = parts.pop() if parts else ""
            for part in parts:
                handle_output_segment(part)
        return consumed

    try:
        while process.poll() is None:
            consume_available_output()
            if pending and not count_output_files:
                percent = extract_transfer_percent(pending)
                if percent is not None and progress_callback and percent != last_callback_progress:
                    progress_callback(percent)
                    last_callback_progress = percent
            time.sleep(0.2)
        consume_available_output()
        while True:
            try:
                chunk = os.read(stdout_fd, 4096)
            except BlockingIOError:
                break
            except OSError as error:
                if error.errno == errno.EIO and use_pty:
                    break
                raise
            if not chunk:
                break
            pending += chunk.decode("utf-8", errors="replace")
            parts = re.split(r"[\r\n]+", pending)
            pending = parts.pop() if parts else ""
            for part in parts:
                handle_output_segment(part)
        if pending:
            handle_output_segment(pending, force=True)
        return process.wait()
    finally:
        if master_fd is not None:
            os.close(master_fd)


def remote_path_exists(resource: dict, target_path: str):
    try:
        import paramiko
        from routes.resources import ssh_connect_kwargs
    except ImportError:
        return False, "当前 Python 环境未安装 paramiko，无法检查远程目录。"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        command = f"test -e {remote_shell_path(target_path)}"
        _, stdout, _ = client.exec_command(command, timeout=10)
        return stdout.channel.recv_exit_status() == 0, ""
    except Exception as error:
        return False, f"检查远程目录失败：{error}"
    finally:
        client.close()


def remote_prepare(resource: dict, target_path: str, overwrite: bool):
    try:
        import paramiko
        from routes.resources import ssh_connect_kwargs
    except ImportError:
        return "当前 Python 环境未安装 paramiko，无法准备远程目录。"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        if overwrite:
            command = f"rm -rf {remote_shell_path(target_path)} && mkdir -p {remote_shell_path(target_path)}"
        else:
            command = f"mkdir -p {remote_shell_path(target_path)}"
        _, stdout, stderr = client.exec_command(command, timeout=60)
        code = stdout.channel.recv_exit_status()
        if code != 0:
            return stderr.read().decode("utf-8", errors="replace") or "远程目录准备失败"
        return ""
    except Exception as error:
        return f"准备远程目录失败：{error}"
    finally:
        client.close()


def remote_delete_path(resource: dict, target_path: str):
    try:
        import paramiko
        from routes.resources import ssh_connect_kwargs
    except ImportError:
        return "当前 Python 环境未安装 paramiko，无法删除远程数据集。"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=resource["host"],
            port=resource["port"],
            username=resource["username"],
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            look_for_keys=False,
            allow_agent=False,
            **ssh_connect_kwargs(resource),
        )
        command = f"rm -rf {remote_shell_path(target_path)}"
        _, stdout, stderr = client.exec_command(command, timeout=60)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error = stderr.read().decode("utf-8", errors="replace").strip()
            return error or f"远程删除失败，退出码 {exit_status}"
        return ""
    except Exception as error:
        return f"删除远程数据集失败：{error}"
    finally:
        client.close()


def deploy_transfer_tool():
    rsync = shutil.which("rsync")
    if rsync:
        return "rsync", rsync
    scp = shutil.which("scp")
    if scp:
        return "scp", scp
    return "", ""


def rsync_file_output_args(rsync_path: str, log_path: Path | None = None):
    version_label = "未知版本"
    help_text = ""
    try:
        version = subprocess.run(
            [rsync_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
        first_line = (version.stdout or "").splitlines()[0].strip()
        if first_line:
            version_label = first_line
    except Exception:
        pass
    try:
        help_result = subprocess.run(
            [rsync_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
        help_text = help_result.stdout or ""
    except Exception:
        help_text = ""
    try:
        result = subprocess.run(
            [rsync_path, "--info=help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except Exception:
        if log_path:
            append_deploy_log(log_path, f"rsync {version_label}，无法检测 --info=progress2，使用文件数量进度。")
        args = ["--out-format=%n"]
        if "--outbuf" in help_text:
            args.insert(0, "--outbuf=L")
        return args
    supports_progress2 = "progress2" in result.stdout
    args = ["--out-format=%n"]
    line_buffered = "--outbuf" in help_text
    if line_buffered:
        args.insert(0, "--outbuf=L")
    if log_path:
        flag = "支持 --info=progress2" if supports_progress2 else "不支持 --info=progress2"
        buffer_flag = "启用行缓冲输出" if line_buffered else "当前 rsync 不支持 --outbuf=L"
        append_deploy_log(log_path, f"rsync {version_label}，{flag}，使用文件数量进度，{buffer_flag}。")
    return args


def build_deploy_commands(task: dict, source: Path, resource: dict | None, log_path: Path):
    target_path = str(task["target_path"]).strip()
    mode = task["mode"]
    temp_files: list[Path] = []
    prefix: list[str] = []
    ssh_key = None
    auth_error = ""
    tool_name, tool_path = deploy_transfer_tool()
    if not tool_name:
        return [], temp_files, "", "本机未安装 rsync 或 scp，无法执行部署。"
    if task["target_type"] == "remote":
        assert resource is not None
        prefix, ssh_key, auth_error = prepare_remote_auth(resource, temp_files)
        if auth_error:
            return [], temp_files, tool_name, auth_error
        target_arg = remote_target(resource, target_path.rstrip("/") + "/")
        if tool_name == "rsync":
            source_arg = str(source) + "/"
            base = [*prefix, tool_path, "-az", *rsync_file_output_args(tool_path, log_path), "-e", rsync_ssh_args(resource, ssh_key)]
        else:
            source_arg = str(source / ".")
            base = [*prefix, tool_path, "-r", "-P", str(resource.get("port") or 22), "-o", "StrictHostKeyChecking=no"]
            if ssh_key:
                base.extend(["-i", str(ssh_key)])
    else:
        target_arg = str(Path(target_path).expanduser()) + "/"
        if tool_name == "rsync":
            source_arg = str(source) + "/"
            base = [tool_path, "-az", *rsync_file_output_args(tool_path, log_path)]
        else:
            source_arg = str(source / ".")
            base = [tool_path, "-r"]

    if mode == "full":
        command = [*base, source_arg, target_arg]
        if tool_name == "rsync":
            command.insert(len(base), "--delete")
        return [command], temp_files, tool_name, ""
    if mode in {"sync", "diff", "incremental"}:
        return [[*base, source_arg, target_arg]], temp_files, tool_name, ""
    append_deploy_log(log_path, f"未知部署方式: {mode}")
    return [], temp_files, tool_name, "未知部署方式"


def run_deploy_task(project_path: Path, task: dict):
    log_path = Path(task["log_path"])
    source = Path(task["source_path"])
    task_id = task["id"]
    temp_files: list[Path] = []
    try:
        update_deploy_task(project_path, task_id, status="进行中", progress=5)
        append_deploy_log(log_path, f"开始部署数据集 {task['dataset']}")
        if not source.is_dir():
            raise RuntimeError("源数据集目录不存在")

        resource = None
        if task["target_type"] == "remote":
            resource = find_resource(workspace_path(), task.get("resource_id", ""))
            if resource is None:
                raise RuntimeError("远程服务器不存在")
            exists, error = remote_path_exists(resource, task["target_path"])
            if error:
                raise RuntimeError(error)
            if task["mode"] == "full" and exists and not task.get("overwrite"):
                message = "目标目录已存在，不能执行全量部署。请确认是否删除覆盖。"
                update_deploy_task(project_path, task_id, status="等待确认", progress=0, error=message)
                append_deploy_log(log_path, message)
                return
            prepare_error = remote_prepare(resource, task["target_path"], task["mode"] == "full" and bool(task.get("overwrite")))
            if prepare_error:
                raise RuntimeError(prepare_error)
        else:
            target = Path(task["target_path"]).expanduser()
            if task["mode"] == "full" and target.exists():
                if not task.get("overwrite"):
                    message = "目标目录已存在，不能执行全量部署。请确认是否删除覆盖。"
                    update_deploy_task(project_path, task_id, status="等待确认", progress=0, error=message)
                    append_deploy_log(log_path, message)
                    return
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

        update_deploy_task(project_path, task_id, progress=20)
        commands, temp_files, tool_name, error = build_deploy_commands(task, source, resource, log_path)
        if error:
            raise RuntimeError(error)
        total_files = count_deploy_files(source)
        append_deploy_log(log_path, f"待复制文件：{total_files} 个")
        for index, command in enumerate(commands, start=1):
            append_deploy_log(log_path, f"执行 {tool_name} ({index}/{len(commands)})")
            last_task_progress = 20
            last_progress_update = 0.0

            def sync_transfer_progress(percent: int):
                nonlocal last_task_progress, last_progress_update
                command_start = 20 + round((index - 1) / len(commands) * 70)
                command_span = 70 / len(commands)
                next_progress = command_start + round(percent / 100 * command_span)
                next_progress = max(last_task_progress, min(next_progress, 90))
                now = time.monotonic()
                if next_progress == last_task_progress and now - last_progress_update < 2:
                    return
                update_deploy_task(project_path, task_id, progress=next_progress)
                last_task_progress = next_progress
                last_progress_update = now

            code = run_command(
                command,
                log_path,
                sync_transfer_progress,
                progress_total=total_files,
                count_output_files=tool_name == "rsync",
                use_pty=tool_name == "rsync",
            )
            if code != 0:
                raise RuntimeError(f"{tool_name} 退出码 {code}")
            update_deploy_task(project_path, task_id, progress=20 + round(index / len(commands) * 70))
        update_deploy_task(project_path, task_id, status="完成", progress=100)
        if total_files and not current_copy_file_from_log(log_path.read_text(encoding="utf-8", errors="replace")):
            append_deploy_log(log_path, "没有文件需要复制")
        append_deploy_log(log_path, "部署完成")
    except Exception as error:
        update_deploy_task(project_path, task_id, status="失败", progress=100, error=str(error))
        append_deploy_log(log_path, f"部署失败：{error}")
    finally:
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def create_deploy_task(project_path: Path, dataset_path: Path, dataset_name: str, form):
    resource_id = (form.get("resource_id", "") or "").strip()
    target_type = "remote"
    mode = (form.get("mode", "sync") or "sync").strip()
    if mode not in DEPLOY_MODES:
        return None, "部署方式不正确"
    resource = find_resource(workspace_path(), resource_id)
    if resource is None:
        return None, "请选择算力服务器"
    target_path = (form.get("target_path", "") or "").strip() or f"~/datasets/{dataset_name}"
    if "\n" in target_path or "\r" in target_path:
        return None, "部署位置不能包含换行"
    task_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    log_path = deploy_root(project_path) / f"{task_id}.log"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task = {
        "id": task_id,
        "dataset": dataset_name,
        "icon": read_dataset_meta(dataset_path)["icon"],
        "source_path": str(dataset_path),
        "target_type": target_type,
        "target_path": target_path,
        "resource_id": resource_id,
        "resource_name": resource.get("name", "") if resource else "",
        "mode": mode,
        "overwrite": False,
        "status": "排队中",
        "progress": 0,
        "error": "",
        "log_path": str(log_path),
        "created_at": now,
        "updated_at": now,
    }
    with deploy_lock:
        tasks = read_deploy_tasks(project_path)
        tasks.insert(0, task)
        write_deploy_tasks(project_path, tasks)
    append_deploy_log(log_path, "部署任务已创建")
    ensure_dataset_worker(workspace_path())
    return task, ""


def zip_dataset(path: Path):
    temp = tempfile.NamedTemporaryFile(prefix=f"{path.name}-", suffix=".zip", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix().lower()):
            if file.is_file():
                archive.write(file, file.relative_to(path).as_posix())
    return temp_path


def split_image_items(path: Path):
    items = {}
    for split in ("train", "val", "test"):
        split_dir = path / "images" / split
        labels_dir = path / "labels" / split
        if not split_dir.is_dir():
            split_dir = path / split
            labels_dir = split_dir
        files = image_files(split_dir)
        items[split] = [
            {
                "name": file.relative_to(split_dir).as_posix(),
                "media": f"/dataset/{path.parent.parent.name}/{path.name}/media/{split}/{file.relative_to(split_dir).as_posix()}",
                "thumb": f"/dataset/{path.parent.parent.name}/{path.name}/thumb/{split}/{file.relative_to(split_dir).as_posix()}",
                "label": (labels_dir / file.relative_to(split_dir).with_suffix(".txt")).is_file(),
            }
            for file in files
        ]
    return items


def dataset_image_path(project: str, name: str, split: str, file_path: str):
    path = dataset_dir(workspace_path(), project, name)
    if path is None or split not in {"train", "val", "test"}:
        return None
    root = (path / "images" / split).resolve()
    if not root.is_dir():
        root = (path / split).resolve()
    image = (root / file_path).resolve()
    if not is_inside(image, root) or not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
        return None
    return image


def jpeg_exif_thumbnail(image: Path):
    if image.suffix.lower() not in JPEG_EXTS:
        return None
    try:
        data = image.read_bytes()
    except OSError:
        return None
    pos = 2 if data.startswith(b"\xff\xd8") else 0
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        pos += 2
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA or pos + 2 > len(data):
            break
        segment_length = int.from_bytes(data[pos:pos + 2], "big")
        segment_start = pos + 2
        segment_end = pos + segment_length
        if marker == 0xE1 and data[segment_start:segment_start + 6] == b"Exif\x00\x00":
            return exif_thumbnail_from_segment(data[segment_start + 6:segment_end])
        pos = segment_end
    return None


def exif_thumbnail_from_segment(tiff: bytes):
    if len(tiff) < 8:
        return None
    endian = tiff[:2]
    if endian == b"II":
        byteorder = "little"
    elif endian == b"MM":
        byteorder = "big"
    else:
        return None
    if int.from_bytes(tiff[2:4], byteorder) != 42:
        return None

    def read_u16(offset: int):
        if offset < 0 or offset + 2 > len(tiff):
            return None
        return int.from_bytes(tiff[offset:offset + 2], byteorder)

    def read_u32(offset: int):
        if offset < 0 or offset + 4 > len(tiff):
            return None
        return int.from_bytes(tiff[offset:offset + 4], byteorder)

    ifd0 = read_u32(4)
    if ifd0 is None:
        return None
    count0 = read_u16(ifd0)
    if count0 is None:
        return None
    next_ifd_offset = ifd0 + 2 + count0 * 12
    ifd1 = read_u32(next_ifd_offset)
    if not ifd1:
        return None
    count1 = read_u16(ifd1)
    if count1 is None:
        return None
    thumb_offset = None
    thumb_length = None
    for index in range(count1):
        entry = ifd1 + 2 + index * 12
        if entry + 12 > len(tiff):
            return None
        tag = read_u16(entry)
        value = read_u32(entry + 8)
        if tag == 0x0201:
            thumb_offset = value
        elif tag == 0x0202:
            thumb_length = value
    if not thumb_offset or not thumb_length:
        return None
    end = thumb_offset + thumb_length
    if end > len(tiff):
        return None
    thumbnail = tiff[thumb_offset:end]
    return thumbnail if thumbnail.startswith(b"\xff\xd8") else None


def read_classes_for_dataset(dataset_path: Path):
    path = dataset_path / "classes.txt"
    if not path.is_file():
        path = project_classes_file(dataset_path.parent.parent) or dataset_path.parent.parent / "classes.txt"
    if not path.is_file():
        return {"exists": False, "class_names": [], "text": ""}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "exists": True,
        "class_names": [line.strip() for line in text.splitlines() if line.strip()],
        "text": text,
    }


def class_annotations(path: Path, class_names: list[str]):
    counts = {}
    labels_root = path / "labels"
    search_root = labels_root if labels_root.is_dir() else path
    for label_file in sorted(search_root.rglob("*.txt"), key=lambda item: item.as_posix().lower()):
        try:
            lines = label_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                index = int(float(parts[0]))
            except ValueError:
                continue
            counts[index] = counts.get(index, 0) + 1

    max_index = max(counts.keys(), default=-1)
    total_classes = max(len(class_names), max_index + 1)
    rows = []
    max_count = max(counts.values(), default=0)
    total_annotations = sum(counts.values())
    for index in range(total_classes):
        count = counts.get(index, 0)
        rows.append(
            {
                "index": index,
                "name": class_names[index] if index < len(class_names) else f"class_{index}",
                "count": count,
                "percent": round(count / max_count * 100, 2) if max_count else 0,
            }
        )
    def scale_label(value: float):
        return str(int(value)) if value.is_integer() else f"{value:.1f}"

    scale = [scale_label(max_count * ratio) for ratio in (1, 0.75, 0.5, 0.25, 0)] if max_count else ["0"]
    return {
        "rows": rows,
        "total_classes": total_classes,
        "total_annotations": total_annotations,
        "total_annotations_label": f"{total_annotations:,}",
        "scale": scale,
    }


def dataset_items(workspace: Path, project: str = ""):
    datasets = []
    if not workspace.is_dir():
        return datasets

    for project_dir in sorted(workspace.iterdir(), key=lambda item: item.name.lower()):
        if project and project_dir.name != project:
            continue
        datasets_dir = project_dir / "datasets"
        active_tasks = active_build_tasks(project_dir) if project_dir.is_dir() else []
        active_names = {str(task.get("name", "")) for task in active_tasks if task.get("status") in {"排队中", "创建中", "失败"}}
        for task in active_tasks:
            created_at = str(task.get("created_at") or task.get("updated_at") or "")
            created_sort = datetime_sort_value(created_at, time.time())
            preview = expected_dataset_preview(project_dir, task)
            datasets.append(
                {
                    "name": task.get("name", ""),
                    "icon": dataset_icon(str(task.get("icon") or "")),
                    "path": project_dir / "datasets" / str(task.get("name", "")),
                    "created_at": created_at,
                    "created_sort": created_sort,
                    "updated_at": created_sort,
                    "updated_date": task.get("updated_at", ""),
                    "project": project_name(project_dir),
                    "location_label": "本地",
                    "project_dir": project_dir.name,
                    "splits": preview["splits"],
                    "total_images": preview["total_images"],
                    "total_labels": preview["total_labels"],
                    "chart_style": preview["chart_style"],
                    "chart_segments": preview["chart_segments"],
                    "building": True,
                    "build_task_id": task.get("id", ""),
                    "build_status": task.get("status", ""),
                    "build_progress": int(task.get("progress") or 0),
                    "build_error": normalize_task_error(str(task.get("error", ""))),
                    "deploying": False,
                    "deploy_progress": 0,
                    "deploy_status": "",
                    "deploy_error": task.get("deploy_error", ""),
                    "current_file": current_copy_file_for_task(task) if task.get("status") in {"排队中", "创建中"} else "",
                }
            )
        if not project_dir.is_dir() or not datasets_dir.is_dir():
            continue
        for dataset_dir in sorted(datasets_dir.iterdir(), key=lambda item: item.name.lower()):
            if not dataset_dir.is_dir() or dataset_dir.name in active_names:
                continue
            meta = read_dataset_meta(dataset_dir)
            fallback_created = dataset_dir.stat().st_ctime
            created_sort = datetime_sort_value(meta.get("created_at", ""), fallback_created)
            splits = {
                "train": count_dataset_split(dataset_dir, "train"),
                "val": count_dataset_split(dataset_dir, "val"),
                "test": count_dataset_split(dataset_dir, "test"),
            }
            chart = dataset_chart_data(splits)
            deploy_task = latest_dataset_deploy_task(project_dir, dataset_dir.name)
            deploy_status = str(deploy_task.get("status") or "") if deploy_task else ""
            deploy_progress = int(deploy_task.get("progress") or 0) if deploy_task else 0
            deploying = bool(deploy_task) and deploy_status in ACTIVE_DEPLOY_STATUSES
            deploy_failed = bool(deploy_task) and deploy_status == "失败"
            current_file = deploy_current_file_for_task(deploy_task) if deploying else ""
            location_label = str(deploy_task.get("resource_name") or "算力服务器") if deploy_task else "本地"
            datasets.append(
                {
                    "name": dataset_dir.name,
                    "icon": meta["icon"],
                    "path": dataset_dir,
                    "created_at": meta.get("created_at", ""),
                    "created_sort": created_sort,
                    "updated_at": created_sort,
                    "updated_date": datetime.fromtimestamp(created_sort).strftime("%Y-%m-%d %H:%M"),
                    "project": project_name(project_dir),
                    "location_label": location_label,
                    "project_dir": project_dir.name,
                    "splits": splits,
                    "total_images": chart["total_images"],
                    "total_labels": chart["total_labels"],
                    "chart_style": chart["chart_style"],
                    "chart_segments": chart["chart_segments"],
                    "building": False,
                    "build_progress": 100,
                    "deploying": deploying,
                    "deploy_failed": deploy_failed,
                    "deploy_task_id": deploy_task.get("id", "") if deploy_task else "",
                    "deploy_progress": 100 if deploy_status == "完成" else deploy_progress,
                    "deploy_status": deploy_status,
                    "deploy_error": deploy_task.get("error", "") if deploy_task else "",
                    "current_file": current_file,
                }
            )
    return sorted(datasets, key=lambda item: (item.get("created_sort", 0), item.get("name", "")), reverse=True)


def dataset_summary(path: Path, project: str, name: str):
    splits = {
        "train": count_dataset_split(path, "train"),
        "val": count_dataset_split(path, "val"),
        "test": count_dataset_split(path, "test"),
    }
    classes = read_classes_for_dataset(path)
    annotations = class_annotations(path, classes["class_names"])
    project_path = path.parent.parent
    deploy_task = latest_dataset_deploy_task(project_path, name)
    return {
        "name": name,
        "icon": read_dataset_meta(path)["icon"],
        "project_dir": project,
        "project": project_name(path.parent.parent),
        "deploy_target": str(deploy_task.get("resource_name") or "") if deploy_task else "",
        "path": path,
        "splits": splits,
        "total_images": sum(split["images"] for split in splits.values()),
        "total_labels": sum(split["labels"] for split in splits.values()),
        "files": split_image_items(path),
        "classes": classes,
        "annotations": annotations,
    }


@router.get("/dataset")
def dataset(request: Request, project: str = ""):
    workspace = workspace_path()
    ensure_dataset_worker(workspace)
    current_project = current_project_from_request(request, project)
    if current_project:
        return RedirectResponse(url=f"/dataset/{current_project}", status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": dataset_items(workspace, ""),
            "active_page": "dataset",
            "current_project": "",
            "current_project_name": "",
            "dataset_icons": DATASET_ICONS,
            "project_classes": project_classes_preview(None),
            "resources": read_resources(workspace),
            **header_context(request, workspace),
        },
    )
    return response


@router.get("/dataset/{project}")
def dataset_with_project(request: Request, project: str):
    workspace = workspace_path()
    ensure_dataset_worker(workspace)
    current_project_path = project_dir(workspace, project)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/index.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": dataset_items(workspace, project),
            "active_page": "dataset",
            "current_project": project,
            "current_project_name": project_name(current_project_path) if current_project_path and current_project_path.is_dir() else project,
            "dataset_icons": DATASET_ICONS,
            "project_classes": project_classes_preview(current_project_path),
            "resources": read_resources(workspace),
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/dataset/{project}/deploy")
@router.get("/dataset/{project}/deploy/{name}")
def dataset_deploy(request: Request, project: str, name: str = ""):
    workspace = workspace_path()
    ensure_dataset_worker(workspace)
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    datasets = dataset_select_items(workspace, project)
    selected = next((item for item in datasets if item["name"] == name), datasets[0] if datasets else None)
    tasks = [deploy_task_view(task) for task in read_deploy_tasks(current_project_path)]
    latest_task = tasks[0] if tasks else None
    response = templates.TemplateResponse(
        request=request,
        name="dataset/deploy.html",
        context={
            "request": request,
            "workspace": workspace,
            "datasets": datasets,
            "selected_dataset": selected,
            "tasks": tasks,
            "resources": read_resources(workspace),
            "active_page": "dataset",
            "current_project": project,
            "current_project_name": project_name(current_project_path),
            "deploy_modes": DEPLOY_MODES,
            "deploy_targets": DEPLOY_TARGETS,
            "footer_console_url": f"/dataset/{project}/deploy/tasks/{latest_task['id']}/log" if latest_task else "",
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.post("/dataset/{project}/deploy")
@router.post("/dataset/{project}/deploy/{name}")
async def create_dataset_deploy(request: Request, project: str, name: str = ""):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    form = await request.form()
    dataset_name = name or str(form.get("dataset", "") or "").strip()
    path = dataset_dir(workspace, project, dataset_name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    task, error = create_deploy_task(current_project_path, path, dataset_name, form)
    if error:
        return JSONResponse({"ok": False, "error": error}, status_code=400)
    return RedirectResponse(
        url=f"/dataset/{project}/deploy/{dataset_name}#task-{task['id']}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/dataset/{project}/{name}/deploy/run")
def run_dataset_deploy_from_card(project: str, name: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    path = dataset_dir(workspace, project, name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    deploy_form = dataset_deploy_defaults(current_project_path, name)
    if deploy_form is None or not deploy_form.get("resource_id"):
        return JSONResponse({"ok": False, "error": "暂无算力服务器，请先在算力页添加。"}, status_code=400)
    task, error = create_deploy_task(current_project_path, path, name, deploy_form)
    if error:
        return JSONResponse({"ok": False, "error": error}, status_code=400)
    return {
        "ok": True,
        "task_id": task["id"],
        "log_url": f"/dataset/{project}/deploy/tasks/{task['id']}/log",
    }


@router.post("/dataset/{project}/deploy/tasks/{task_id}/overwrite")
def dataset_deploy_task_overwrite(project: str, task_id: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    tasks = read_deploy_tasks(current_project_path)
    task = next((item for item in tasks if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "部署任务不存在"}, status_code=404)
    if task.get("status") != "等待确认" or task.get("mode") != "full":
        return RedirectResponse(url=f"/dataset/{project}/deploy/{task.get('dataset', '')}", status_code=status.HTTP_303_SEE_OTHER)
    task["overwrite"] = True
    task["status"] = "排队中"
    task["progress"] = 0
    task["error"] = ""
    task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with deploy_lock:
        next_tasks = read_deploy_tasks(current_project_path)
        for item in next_tasks:
            if item.get("id") == task_id:
                item.update(task)
                break
        write_deploy_tasks(current_project_path, next_tasks)
    append_deploy_log(Path(task["log_path"]), "已确认删除覆盖，继续部署")
    ensure_dataset_worker(workspace)
    return RedirectResponse(url=f"/dataset/{project}/deploy/{task.get('dataset', '')}#task-{task_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/dataset/{project}/deploy/tasks/{task_id}/retry")
def dataset_deploy_task_retry(project: str, task_id: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    tasks = read_deploy_tasks(current_project_path)
    task = next((item for item in tasks if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "部署任务不存在"}, status_code=404)
    if task.get("status") != "失败":
        return RedirectResponse(url=f"/dataset/{project}/deploy/{task.get('dataset', '')}#task-{task_id}", status_code=status.HTTP_303_SEE_OTHER)
    task["status"] = "排队中"
    task["progress"] = 0
    task["error"] = ""
    task["overwrite"] = False
    task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with deploy_lock:
        next_tasks = read_deploy_tasks(current_project_path)
        for item in next_tasks:
            if item.get("id") == task_id:
                item.update(task)
                break
        write_deploy_tasks(current_project_path, next_tasks)
    append_deploy_log(Path(task["log_path"]), "失败任务重试")
    ensure_dataset_worker(workspace)
    return RedirectResponse(url=f"/dataset/{project}/deploy/{task.get('dataset', '')}#task-{task_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/dataset/{project}/deploy/tasks/{task_id}/delete")
def dataset_deploy_task_delete(project: str, task_id: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    tasks = read_deploy_tasks(current_project_path)
    task = next((item for item in tasks if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "部署任务不存在"}, status_code=404)
    if task.get("status") in ACTIVE_DEPLOY_STATUSES:
        return JSONResponse({"ok": False, "error": "部署任务仍在进行，不能删除。"}, status_code=400)
    with deploy_lock:
        next_tasks = [item for item in read_deploy_tasks(current_project_path) if item.get("id") != task_id]
        write_deploy_tasks(current_project_path, next_tasks)
    return {"ok": True}


@router.get("/dataset/{project}/deploy/tasks/{task_id}/log")
def dataset_deploy_task_log(project: str, task_id: str, offset: int = 0):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    task = next((item for item in read_deploy_tasks(current_project_path) if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "部署任务不存在"}, status_code=404)
    log_path = Path(task.get("log_path", ""))
    size = log_path.stat().st_size if log_path.is_file() else 0
    start = max(0, min(int(offset or 0), size))
    full_log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    log_text = ""
    if log_path.is_file():
        with log_path.open("rb") as handle:
            handle.seek(start)
            log_text = handle.read().decode("utf-8", errors="replace")
    task_status = task.get("status", "")
    task_progress = 100 if task_status == "完成" else task.get("progress", 0)
    current_file = deploy_current_file_label(full_log_text, task_status)
    return {
        "ok": True,
        "status": task_status,
        "progress": task_progress,
        "error": normalize_task_error(str(task.get("error", ""))),
        "log": normalize_console_log(log_text),
        "offset": start,
        "size": size,
        "current_file": current_file,
    }


@router.get("/dataset/{project}/build/tasks/{task_id}/log")
def dataset_build_task_log(project: str, task_id: str, offset: int = 0):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    task = next((item for item in read_build_tasks(current_project_path) if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "创建任务不存在"}, status_code=404)
    log_path = Path(task.get("log_path", ""))
    size = log_path.stat().st_size if log_path.is_file() else 0
    start = max(0, min(int(offset or 0), size))
    full_log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    log_text = ""
    if log_path.is_file():
        with log_path.open("rb") as handle:
            handle.seek(start)
            log_text = handle.read().decode("utf-8", errors="replace")
    task_status = task.get("status", "")
    task_progress = 100 if task_status == "完成" else task.get("progress", 0)
    current_file = current_copy_file_from_log(full_log_text) if task_status in {"排队中", "创建中"} else ""
    deploy_task_id = str(task.get("deploy_task_id") or "")
    return {
        "ok": True,
        "status": task_status,
        "progress": task_progress,
        "error": task.get("error", ""),
        "log": normalize_console_log(log_text),
        "offset": start,
        "size": size,
        "current_file": current_file,
        "deploy_task_id": deploy_task_id,
        "deploy_log_url": f"/dataset/{project}/deploy/tasks/{deploy_task_id}/log" if deploy_task_id else "",
    }


@router.post("/dataset/{project}/build/tasks/{task_id}/retry")
def dataset_build_task_retry(project: str, task_id: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    if current_project_path is None or not current_project_path.is_dir():
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    task = next((item for item in read_build_tasks(current_project_path) if item.get("id") == task_id), None)
    if task is None:
        return JSONResponse({"ok": False, "error": "创建任务不存在"}, status_code=404)
    if task.get("status") != "失败":
        return JSONResponse({"ok": False, "error": "只有失败的创建任务可以重新创建"}, status_code=400)
    dataset_path = current_project_path / "datasets" / str(task.get("name") or "")
    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)
    task["status"] = "排队中"
    task["progress"] = 0
    task["error"] = ""
    task["deploy_error"] = ""
    task["deploy_task_id"] = ""
    task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not task.get("log_path"):
        task["log_path"] = str(build_logs_root(current_project_path) / f"{task_id}.log")
    with build_lock:
        next_tasks = read_build_tasks(current_project_path)
        for item in next_tasks:
            if item.get("id") == task_id:
                item.update(task)
                break
        write_build_tasks(current_project_path, next_tasks)
    append_deploy_log(Path(task["log_path"]), "重新创建数据集")
    ensure_dataset_worker(workspace)
    return {"ok": True}


@router.post("/dataset/{project}/{name}/delete")
async def delete_dataset(request: Request, project: str, name: str):
    workspace = workspace_path()
    current_project_path = project_dir(workspace, project)
    path = dataset_dir(workspace, project, name)
    if current_project_path is None or not current_project_path.is_dir() or path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    payload = await request.json()
    delete_remote = bool(payload.get("delete_remote"))
    deploy_tasks = [task for task in read_deploy_tasks(current_project_path) if task.get("dataset") == name]
    if delete_remote:
        for task in deploy_tasks:
            if task.get("target_type") != "remote":
                continue
            resource = find_resource(workspace, str(task.get("resource_id") or ""))
            target_path = str(task.get("target_path") or "")
            if resource is None or not target_path:
                return JSONResponse({"ok": False, "error": "远程部署记录缺少算力服务器或部署路径"}, status_code=400)
            error = remote_delete_path(resource, target_path)
            if error:
                return JSONResponse({"ok": False, "error": error}, status_code=400)
    shutil.rmtree(path)
    with build_lock:
        build_tasks = [task for task in read_build_tasks(current_project_path) if task.get("name") != name]
        write_build_tasks(current_project_path, build_tasks)
    with deploy_lock:
        next_deploy_tasks = [task for task in read_deploy_tasks(current_project_path) if task.get("dataset") != name]
        write_deploy_tasks(current_project_path, next_deploy_tasks)
    return {"ok": True}


@router.get("/dataset/{project}/{name}")
def dataset_detail(request: Request, project: str, name: str):
    workspace = workspace_path()
    path = dataset_dir(workspace, project, name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    response = templates.TemplateResponse(
        request=request,
        name="dataset/detail.html",
        context={
            "request": request,
            "workspace": workspace,
            "dataset": dataset_summary(path, project, name),
            "active_page": "dataset",
            "current_project": project,
            **header_context(request, workspace),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/dataset/{project}/{name}/media/{split}/{file_path:path}")
def dataset_media(project: str, name: str, split: str, file_path: str):
    image = dataset_image_path(project, name, split, file_path)
    if image is None:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    return FileResponse(image)


@router.get("/dataset/{project}/{name}/thumb/{split}/{file_path:path}")
def dataset_thumb(project: str, name: str, split: str, file_path: str):
    image = dataset_image_path(project, name, split, file_path)
    if image is None:
        return JSONResponse({"ok": False, "error": "图片不存在"}, status_code=404)
    thumbnail = jpeg_exif_thumbnail(image)
    if thumbnail:
        return Response(content=thumbnail, media_type="image/jpeg")
    return FileResponse(image)


@router.post("/dataset")
async def create_dataset(request: Request):
    workspace = workspace_path()
    try:
        payload = await request.json()
        current_project = current_project_from_request(request, str(payload.get("project", "")))
        if not current_project:
            return JSONResponse({"ok": False, "error": "请先进入项目"}, status_code=400)
        current_project_path = project_dir(workspace, current_project)
        if current_project_path is None or not current_project_path.is_dir():
            return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=400)
        task, error = create_build_task(
            current_project_path,
            str(payload.get("name", "")),
            int(payload.get("val_percent", 0) or 0),
            int(payload.get("test_percent", 0) or 0),
            str(payload.get("icon", "")),
            bool(payload.get("deploy_enabled")),
            str(payload.get("deploy_mode", "sync") or "sync"),
            str(payload.get("resource_id", "") or ""),
        )
        if error:
            return JSONResponse({"ok": False, "error": error}, status_code=400)
        return {
            "ok": True,
            "task": task,
            "log_url": f"/dataset/{current_project}/build/tasks/{task['id']}/log",
        }
    except Exception:
        return JSONResponse(
            {"ok": False, "error": "创建数据集失败"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/dataset/{project}/{name}/download")
def download_dataset(project: str, name: str):
    path = dataset_dir(workspace_path(), project, name)
    if path is None:
        return JSONResponse({"ok": False, "error": "数据集不存在"}, status_code=404)
    zip_path = zip_dataset(path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{name}.zip",
        background=BackgroundTask(lambda file: Path(file).unlink(missing_ok=True), zip_path),
    )
