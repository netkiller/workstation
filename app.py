import base64
import binascii
import os
import secrets
import threading
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from routes.annotate import create_workstation, register_annotate_routes
from routes.api import create_workstation_api_router
from routes.dataset import jpeg_exif_thumbnail, router as dataset_router
from routes.help import router as help_router
from routes.model import router as model_router
from routes.project import (
    ANNOTATE_DIR,
    IMAGE_EXTS,
    TEST_DIR,
    build_project_index,
    current_username,
    is_inside,
    project_annotate_workspace,
    project_classify_workspace,
    project_root_workspace,
    router as project_router,
    team_mode_enabled,
    workspace_path,
)
from routes.predict import router as predict_router
from routes.resources import router as resources_router
from routes.test import (
    ensure_worker as ensure_test_worker,
    load_tasks as load_test_tasks,
    prediction_image_path,
    read_report as read_test_report,
    report_model_run_dir,
    report_row_image_name,
    router as test_router,
    test_images_dir,
)
from routes.train import router as train_router
from routes.val import router as validate_router


STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=TEMPLATE_DIR)
MEDIA_CONVERT_EXTS = {".heic", ".heif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}

try:
    from PIL import Image, ImageOps

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pillow_heif = None
except ImportError:
    Image = None
    pillow_heif = None

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(dataset_router)
app.include_router(help_router)
app.include_router(project_router)
app.include_router(predict_router)
app.include_router(resources_router)
app.include_router(test_router)
app.include_router(train_router)
app.include_router(validate_router)
app.include_router(model_router)

_INDEXER_STARTED = False
_INDEXER_INTERVAL = 5


def _dir_signature(path: Path):
    if not path.exists():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    signature = [stat.st_mtime_ns]
    if path.is_dir():
        try:
            for child in path.iterdir():
                if child.is_dir():
                    try:
                        signature.append(child.stat().st_mtime_ns)
                    except OSError:
                        continue
        except OSError:
            pass
    return tuple(signature)


def _project_signature(path: Path):
    return (
        _dir_signature(path / ANNOTATE_DIR),
        _dir_signature(path / TEST_DIR),
        _dir_signature(path / "models"),
    )


def _project_indexer_loop():
    seen = {}
    while True:
        workspace = workspace_path()
        if workspace.is_dir():
            for project in workspace.iterdir():
                if not project.is_dir() or project.name.startswith("."):
                    continue
                signature = _project_signature(project)
                if signature == seen.get(project):
                    continue
                seen[project] = signature
                try:
                    build_project_index(project)
                except Exception:
                    continue
        time.sleep(_INDEXER_INTERVAL)


@app.on_event("startup")
def start_project_indexer():
    global _INDEXER_STARTED
    if _INDEXER_STARTED:
        return
    _INDEXER_STARTED = True
    threading.Thread(target=_project_indexer_loop, daemon=True, name="project-indexer").start()
    ensure_test_worker()


def basic_auth_credentials():
    auth = os.environ.get("YOLOUTILS_AUTH", "").strip()
    if not auth:
        return None
    username, separator, password = auth.partition(":")
    if not separator or not username or not password:
        return None
    return username, password


def unauthorized_response():
    return Response(
        "Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Yolo Workstation", charset="UTF-8"'},
    )


@app.middleware("http")
async def require_basic_auth(request: Request, call_next):
    credentials = basic_auth_credentials()
    if not credentials:
        return await call_next(request)

    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return unauthorized_response()
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return unauthorized_response()
    username, separator, password = decoded.partition(":")
    if not separator:
        return unauthorized_response()
    expected_username, expected_password = credentials
    if not (
        secrets.compare_digest(username, expected_username)
        and secrets.compare_digest(password, expected_password)
    ):
        return unauthorized_response()
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index(request: Request):
    if team_mode_enabled():
        if not current_username(request, workspace_path()):
            return RedirectResponse(url="/team/login")
        return RedirectResponse(url="/team")
    return RedirectResponse(url="/project")


@app.get("/annotate", include_in_schema=False)
def annotate(request: Request):
    project = request.query_params.get("project")
    if project:
        return RedirectResponse(url=f"/detect/{project}")
    return RedirectResponse(url="/detect/")


@app.get("/detect", include_in_schema=False)
def detect(request: Request):
    project = request.query_params.get("project")
    if project:
        return RedirectResponse(url=f"/detect/{project}")
    return RedirectResponse(url="/detect/")


@app.get("/classify", include_in_schema=False)
def classify(request: Request):
    project = request.query_params.get("project")
    if project:
        return RedirectResponse(url=f"/classify/{project}")
    return RedirectResponse(url="/classify/")


def _media_workspace(request: Request, project: str = ""):
    requested_project = project or request.query_params.get("project") or request.cookies.get("current_project", "")
    if requested_project:
        if request.url.path.startswith("/detect/media") or request.url.path.startswith("/annotate/media"):
            annotate_workspace = project_annotate_workspace(requested_project)
            if annotate_workspace is not None:
                return annotate_workspace
        if request.url.path.startswith("/classify/media"):
            classify_workspace = project_classify_workspace(requested_project)
            if classify_workspace is not None:
                return classify_workspace
        project_workspace = project_root_workspace(requested_project)
        if project_workspace is not None:
            return project_workspace
    return workspace_path().resolve()


def _media_path(request: Request, path: str, project: str = ""):
    workspace = _media_workspace(request, project)
    path_parts = [part for part in Path(path or "").parts if part not in ("", ".")]
    if request.url.path == "/media" and len(path_parts) >= 2 and path_parts[0] == TEST_DIR and path_parts[1] == "tasks":
        raise HTTPException(status_code=404, detail="image not found")
    file_path = (workspace / (path or "")).resolve()
    if file_path != workspace and not is_inside(file_path, workspace):
        raise HTTPException(status_code=400, detail="invalid path")
    if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=404, detail="image not found")
    return file_path


def _media_image_response(path: Path):
    if path.suffix.lower() not in MEDIA_CONVERT_EXTS:
        return FileResponse(path)
    if Image is None:
        raise HTTPException(status_code=415, detail="图片转换需要安装 Pillow")
    if path.suffix.lower() in (".heic", ".heif") and pillow_heif is None:
        raise HTTPException(status_code=415, detail="HEIC/HEIF 需要安装 pillow-heif")
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                rgba_image = image.convert("RGBA")
                canvas.paste(rgba_image, mask=rgba_image.getchannel("A"))
                image = canvas
            else:
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=92, optimize=True)
    except Exception as error:
        raise HTTPException(status_code=415, detail=f"图片转换失败: {error}") from error
    return Response(buffer.getvalue(), media_type="image/jpeg")


def _label_indices(label_file: Path):
    indices = set()
    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return indices
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            label_index = int(parts[0])
            [float(value) for value in parts[1:]]
        except ValueError:
            continue
        indices.add(label_index)
    return indices


def _browser_roots(project: str = ""):
    if project:
        project_workspace = project_root_workspace(project)
        if project_workspace is None:
            raise HTTPException(status_code=404, detail="project not found")
        project_workspace = project_workspace.resolve()
        annotate_workspace = (project_workspace / ANNOTATE_DIR).resolve()
        return project_workspace, annotate_workspace
    workspace = workspace_path().resolve()
    return workspace, workspace


def _media_source_url(path: str, project: str = "", *, source: str = "/media", variant: str = ""):
    if source == "/media" and project and variant in {"thumbnail", "original"}:
        return f"/media/{quote(project, safe='')}/{variant}?{urlencode({'path': path})}"
    query = {"path": path}
    if project:
        query["project"] = project
    if variant:
        query["variant"] = variant
    return f"{source}?{urlencode(query)}"


def _image_orientation(path: Path):
    if Image is None:
        return "landscape"
    try:
        with Image.open(path) as image:
            width, height = image.size
            orientation = None
            try:
                orientation = image.getexif().get(274)
            except Exception:
                orientation = None
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
    except Exception:
        return "landscape"
    return "portrait" if height > width else "landscape"


def _media_browser_items(
    root: Path,
    scan_root: Path,
    extensions: set[str],
    project: str = "",
    labels: set[int] | None = None,
):
    if not scan_root.is_dir():
        return []
    items = []
    candidates = []
    for item in scan_root.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in extensions:
            continue
        try:
            created_at = item.stat().st_ctime
        except OSError:
            created_at = 0
        candidates.append((created_at, item))
    for _created_at, item in sorted(candidates, key=lambda value: (-value[0], value[1].as_posix().lower())):
        if labels and not (labels & _label_indices(item.with_suffix(".txt"))):
            continue
        resolved = item.resolve()
        if not is_inside(resolved, root):
            continue
        relative_path = resolved.relative_to(root).as_posix()
        items.append(
            {
                "name": item.name,
                "path": relative_path,
                "src": _media_source_url(
                    relative_path,
                    project,
                    variant="thumbnail" if extensions == IMAGE_EXTS else "",
                ),
                "original_src": _media_source_url(relative_path, project) if extensions == IMAGE_EXTS else "",
                "orientation": _image_orientation(resolved) if extensions == IMAGE_EXTS else "landscape",
            }
        )
    return items


def _media_video_path(request: Request, path: str, project: str = ""):
    workspace = _media_workspace(request, project)
    file_path = (workspace / (path or "")).resolve()
    if file_path != workspace and not is_inside(file_path, workspace):
        raise HTTPException(status_code=400, detail="invalid path")
    if not file_path.is_file() or file_path.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(status_code=404, detail="video not found")
    return file_path


def _report_media_path(task_id: str, image_index: int, model: int | None = None):
    report = read_test_report(task_id)
    task = next((item for item in load_test_tasks() if item.get("id") == task_id), None)
    if report is None or task is None:
        raise HTTPException(status_code=404, detail="report not found")
    project = str(task.get("project") or report.get("project") or "")
    project_workspace = project_root_workspace(project)
    if project_workspace is None:
        raise HTTPException(status_code=404, detail="project not found")
    image_name = report_row_image_name(report, image_index)
    if model is None:
        images_root = test_images_dir(project_workspace)
        selected_sets = []
        if task.get("test_set"):
            selected_sets.append(str(task.get("test_set")))
        selected_sets.extend(str(item) for item in (task.get("test_sets") or []) if str(item))
        candidates = [images_root / image_name]
        for set_name in dict.fromkeys(selected_sets):
            candidates.append(images_root / set_name / image_name)
            candidates.append(images_root / set_name / Path(image_name).name)
        candidates.append(images_root / Path(image_name).name)
        image_path = next(
            (candidate for candidate in candidates if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS),
            None,
        )
        if image_path is None:
            image_path = next(
                (
                    candidate
                    for candidate in sorted(images_root.rglob(Path(image_name).name), key=lambda item: item.as_posix().lower())
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS
                ),
                None,
            )
        if image_path is None:
            raise HTTPException(status_code=404, detail="image not found")
    else:
        models = report.get("models") or []
        if model < 0 or model >= len(models):
            raise HTTPException(status_code=404, detail="model not found")
        run_dir = report_model_run_dir(project_workspace, report, models[model])
        image_path = prediction_image_path(run_dir, image_name) if run_dir else None
        if image_path is None:
            raise HTTPException(status_code=404, detail="image not found")
    resolved = image_path.resolve()
    if not is_inside(resolved, project_workspace.resolve()):
        raise HTTPException(status_code=400, detail="invalid image")
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=404, detail="image not found")
    return resolved


def _thumbnail_response(path: Path):
    thumbnail = jpeg_exif_thumbnail(path)
    if thumbnail:
        return Response(content=thumbnail, media_type="image/jpeg")
    if Image is None:
        return FileResponse(path)
    try:
        with Image.open(path) as image:
            thumbnail = image.info.get("thumbnail")
            if isinstance(thumbnail, bytes) and thumbnail:
                image = Image.open(BytesIO(thumbnail))
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                rgba_image = image.convert("RGBA")
                canvas.paste(rgba_image, mask=rgba_image.getchannel("A"))
                image = canvas
            else:
                image = image.convert("RGB")
            image.thumbnail((520, 520))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=82, optimize=True)
    except Exception as error:
        raise HTTPException(status_code=415, detail=f"缩略图生成失败: {error}") from error
    return Response(buffer.getvalue(), media_type="image/jpeg")


@app.get("/media/original/reports/{task_id}/{image_index}")
def media_report_original(task_id: str, image_index: int, model: int | None = Query(default=None)):
    return _media_image_response(_report_media_path(task_id, image_index, model))


@app.get("/media/thumbnail/reports/{task_id}/{image_index}")
def media_report_thumbnail(task_id: str, image_index: int, model: int | None = Query(default=None)):
    return _thumbnail_response(_report_media_path(task_id, image_index, model))


@app.get("/media/image", include_in_schema=False)
def media_image_browser(
    request: Request,
    project: str = "",
    label: int | None = Query(default=None),
    labels: list[int] | None = Query(default=None),
    label_name: str = "",
    scope: str = "",
    set_name: str = Query(default="", alias="set"),
    return_to: str = "",
):
    root, scan_root = _browser_roots(project)
    if project and scope == "test":
        scan_root = (root / TEST_DIR / "images" / set_name).resolve() if set_name else (root / TEST_DIR / "images").resolve()
    selected_labels = []
    if label is not None:
        selected_labels.append(label)
    selected_labels.extend(labels or [])
    selected_label_set = set(selected_labels)
    items = _media_browser_items(root, scan_root, IMAGE_EXTS, project=project, labels=selected_label_set)
    subtitle_parts = []
    if project:
        subtitle_parts.append(f"项目 {project}")
    if scope == "test":
        subtitle_parts.append(f"测试集 {set_name}" if set_name else "测试集")
    if selected_labels:
        label_text = f"{label}: {label_name}" if label is not None and label_name else ", ".join(str(item) for item in selected_labels)
        subtitle_parts.append(f"标签 {label_text}")
    return templates.TemplateResponse(
        request=request,
        name="media/browser.html",
        context={
            "request": request,
            "mode": "image",
            "title": "图像浏览器",
            "subtitle": " / ".join(subtitle_parts) or "全部图像",
            "project": project,
            "count": len(items),
            "items": items,
            "return_to": return_to,
        },
    )


@app.get("/media/video", include_in_schema=False)
def media_video_browser(request: Request, project: str = ""):
    root, scan_root = _browser_roots(project)
    items = _media_browser_items(root, scan_root, VIDEO_EXTS, project=project)
    for item in items:
        item["src"] = _media_source_url(item["path"], project, source="/media/video/source")
    return templates.TemplateResponse(
        request=request,
        name="media/browser.html",
        context={
            "request": request,
            "mode": "video",
            "title": "视频浏览器",
            "subtitle": f"项目 {project}" if project else "全部视频",
            "project": project,
            "count": len(items),
            "items": items,
        },
    )


@app.get("/media/video/source", include_in_schema=False)
def media_video_source(request: Request, path: str, project: str = ""):
    return FileResponse(_media_video_path(request, path, project))


@app.get("/media/{project}/thumbnail", include_in_schema=False)
def media_project_thumbnail(request: Request, project: str, path: str):
    return _thumbnail_response(_media_path(request, path, project))


@app.get("/media/{project}/original", include_in_schema=False)
def media_project_original(request: Request, project: str, path: str):
    return _media_image_response(_media_path(request, path, project))


@app.get("/media")
@app.get("/detect/media")
@app.get("/annotate/media")
@app.get("/classify/media")
def media(
    request: Request,
    path: str = "",
    project: str = "",
    raw: bool = Query(default=False),
    report: str = "",
    image: int | None = Query(default=None),
    model: int | None = Query(default=None),
    variant: str = Query(default="original"),
):
    if report:
        if image is None:
            raise HTTPException(status_code=400, detail="image index required")
        report_path = _report_media_path(report, image, model)
        if variant == "thumbnail":
            return _thumbnail_response(report_path)
        if variant not in ("original", "raw"):
            raise HTTPException(status_code=400, detail="invalid media variant")
        return _media_image_response(report_path)
    if not path:
        raise HTTPException(status_code=404, detail="image not found")
    file_path = _media_path(request, path, project)
    if variant == "thumbnail":
        return _thumbnail_response(file_path)
    if raw:
        return FileResponse(file_path)
    return _media_image_response(file_path)


def create_annotate_application():
    workstation = create_workstation()
    annotate_app = FastAPI(title="Yolo Workstation")
    annotate_app.include_router(create_workstation_api_router(workstation))
    return register_annotate_routes(annotate_app, workstation)


def create_detect_application():
    workstation = create_workstation()
    detect_app = FastAPI(title="Yolo Workstation Detect")
    detect_app.include_router(create_workstation_api_router(workstation))
    return register_annotate_routes(
        detect_app,
        workstation,
        route_prefix="detect",
        template_section="annotate",
        workspace_getter=project_annotate_workspace,
        active_mode="annotate",
        mode_label="标注",
        mode_icon="▧",
        error_label="Detect",
    )


def create_classify_application():
    workstation = create_workstation()
    classify_app = FastAPI(title="Yolo Workstation Classify")
    classify_app.include_router(create_workstation_api_router(workstation))
    return register_annotate_routes(
        classify_app,
        workstation,
        route_prefix="classify",
        template_section="classify",
        workspace_getter=project_classify_workspace,
        active_mode="classify",
        mode_label="分类",
        mode_icon="▨",
        error_label="Classify",
    )


app.mount("/detect", create_detect_application())
app.mount("/annotate", create_annotate_application())
app.mount("/classify", create_classify_application())


def run(host: str | None = None, port: int | None = None, reload: bool = False):
    import uvicorn

    host = host or os.environ.get("YOLOUTILS_HOST", "0.0.0.0")
    try:
        port = int(port if port is not None else os.environ.get("YOLOUTILS_PORT", "8000"))
    except (TypeError, ValueError):
        port = 8000
    uvicorn.run("app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run()
