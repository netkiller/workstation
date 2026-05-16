import base64
import binascii
import os
import secrets
import threading
import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from routes.annotate import create_workstation, register_annotate_routes
from routes.api import create_workstation_api_router
from routes.dataset import router as dataset_router
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
    router as project_router,
    team_mode_enabled,
    workspace_path,
)
from routes.predict import router as predict_router
from routes.resources import router as resources_router
from routes.test import ensure_worker as ensure_test_worker, router as test_router
from routes.train import router as train_router
from routes.val import router as validate_router


STATIC_DIR = Path(__file__).resolve().parent / "static"
MEDIA_CONVERT_EXTS = {".heic", ".heif", ".tif", ".tiff"}

try:
    from PIL import Image

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
            return RedirectResponse(url="/login")
        return RedirectResponse(url="/team")
    return RedirectResponse(url="/project")


@app.get("/annotate", include_in_schema=False)
def annotate(request: Request):
    project = request.query_params.get("project")
    if project:
        return RedirectResponse(url=f"/annotate/{project}")
    return RedirectResponse(url="/annotate/")


def _media_workspace(request: Request, project: str = ""):
    requested_project = project or request.query_params.get("project") or request.cookies.get("current_project", "")
    if requested_project:
        annotate_workspace = project_annotate_workspace(requested_project)
        if annotate_workspace is not None:
            return annotate_workspace
    return workspace_path().resolve()


def _media_path(request: Request, path: str, project: str = ""):
    workspace = _media_workspace(request, project)
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


@app.get("/media")
@app.get("/annotate/media")
def media(
    request: Request,
    path: str,
    project: str = "",
    raw: bool = Query(default=False),
):
    file_path = _media_path(request, path, project)
    if raw:
        return FileResponse(file_path)
    return _media_image_response(file_path)


def create_annotate_application():
    workstation = create_workstation()
    annotate_app = FastAPI(title="Yolo Workstation")
    annotate_app.include_router(create_workstation_api_router(workstation))
    return register_annotate_routes(annotate_app, workstation)


app.mount("/annotate", create_annotate_application())


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
