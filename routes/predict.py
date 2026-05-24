import shutil
import posixpath
import shlex
import stat
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from routes.project import header_context
from routes.resources import find_resource, read_resources, ssh_connect_kwargs
from routes.train import load_tasks as load_train_tasks, model_items as train_model_items
from routes.val import project_path, read_project_name, run_model_items as val_model_items, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
MODEL_CACHE = {}


def is_inside(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def predict_root(path: Path):
    root = path / "predict"
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    return root


def safe_upload_name(filename: str):
    raw = Path((filename or "").replace("\\", "/")).name
    stem = Path(raw).stem.strip() or "media"
    suffix = Path(raw).suffix.lower()
    if suffix not in MEDIA_EXTS:
        return ""
    allowed = []
    for char in stem:
        allowed.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(allowed)[:80] + suffix


def resolve_model(path: Path, model: str):
    model_path = (path / model).resolve()
    if not model_path.is_file() or not is_inside(model_path, path):
        return None
    return model_path


def default_model_value(path: Path, project: str):
    models = predict_model_items(path, project)
    return str(models[0].get("relative_path") or "") if models else ""


def load_yolo_model(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("缺少 ultralytics，无法执行手机预测。") from error
    key = str(model_path)
    model = MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(key)
        MODEL_CACHE[key] = model
    return model


def predict_image_boxes(model_path: Path, image_path: Path):
    model = load_yolo_model(model_path)
    results = model.predict(str(image_path), verbose=False)
    boxes = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is None or result_boxes.xyxy is None:
            continue
        coords = result_boxes.xyxy.cpu().tolist()
        classes = result_boxes.cls.cpu().tolist() if result_boxes.cls is not None else [0] * len(coords)
        confidences = result_boxes.conf.cpu().tolist() if result_boxes.conf is not None else [0] * len(coords)
        height, width = getattr(result, "orig_shape", (0, 0))[:2]
        if not width or not height:
            continue
        for index, xyxy in enumerate(coords):
            class_id = int(classes[index])
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            boxes.append(
                {
                    "label": label,
                    "confidence": round(float(confidences[index]), 4),
                    "x1": float(xyxy[0]) / width,
                    "y1": float(xyxy[1]) / height,
                    "x2": float(xyxy[2]) / width,
                    "y2": float(xyxy[3]) / height,
                }
            )
    return boxes


async def save_upload_files(path: Path, files: list[UploadFile], capture_kind: str = ""):
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    upload_dir = predict_root(path) / "uploads" / run_name
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, file in enumerate(files, start=1):
        name = safe_upload_name(file.filename or f"{capture_kind or 'media'}_{index}.jpg")
        if not name:
            continue
        target = upload_dir / name
        content = await file.read()
        if not content:
            continue
        target.write_bytes(content)
        saved.append(target)
    return run_name, upload_dir, saved


def output_items(project: str, project_dir: Path, run_dir: Path, model_name: str = ""):
    items = []
    if not run_dir.is_dir():
        return items
    for file in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not file.is_file() or file.suffix.lower() not in MEDIA_EXTS:
            continue
        relative = file.relative_to(project_dir).as_posix()
        items.append(
            {
                "name": file.name,
                "model": model_name,
                "type": "video" if file.suffix.lower() in VIDEO_EXTS else "image",
                "url": f"/model/{project}/predict/files/{relative}",
            }
        )
    return items


def run_predict(project: str, path: Path, model_path: Path, upload_dir: Path, run_name: str, model_name: str = ""):
    run_dir = path / "predict-runs" / run_name
    if shutil.which("yolo") is None:
        return {
            "ok": False,
            "error": "yolo 命令不存在，请先安装 ultralytics。",
            "command": "",
            "outputs": [],
        }
    command = [
        "yolo",
        "detect",
        "predict",
        f"model={model_path}",
        f"source={upload_dir}",
        f"project={path / 'predict-runs'}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
    ]
    result = subprocess.run(
        command,
        cwd=workspace_path(),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": result.returncode == 0,
        "error": "" if result.returncode == 0 else (result.stderr or result.stdout or "预测失败"),
        "command": " ".join(str(part) for part in command),
        "outputs": output_items(project, path, run_dir, model_name),
    }


def sftp_mkdirs(sftp, path: str):
    current = "/" if path.startswith("/") else "."
    for part in [item for item in path.split("/") if item]:
        current = posixpath.join(current, part) if current != "/" else f"/{part}"
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def sftp_upload_file(sftp, source: Path, target: str):
    sftp_mkdirs(sftp, posixpath.dirname(target))
    sftp.put(str(source), target)


def sftp_upload_tree(sftp, source_dir: Path, remote_dir: str):
    for item in source_dir.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source_dir).as_posix()
        sftp_upload_file(sftp, item, posixpath.join(remote_dir, relative))


def sftp_download_tree(sftp, remote_dir: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_child = posixpath.join(remote_dir, entry.filename)
        local_child = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            sftp_download_tree(sftp, remote_child, local_child)
        else:
            local_child.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_child, str(local_child))


def run_remote_predict(
    project: str,
    path: Path,
    resource: dict,
    model_path: Path,
    upload_dir: Path,
    run_name: str,
    model_name: str = "",
):
    try:
        import paramiko
    except ImportError:
        return {
            "ok": False,
            "error": "当前 Python 环境未安装 paramiko，无法连接远程算力。",
            "command": "",
            "outputs": [],
        }

    remote_root = f"/tmp/yolows_predict_{run_name}"
    remote_source = posixpath.join(remote_root, "source")
    remote_model = posixpath.join(remote_root, "model", model_path.name)
    remote_project = posixpath.join(remote_root, "runs")
    remote_run_dir = posixpath.join(remote_project, run_name)
    local_run_dir = path / "predict-runs" / run_name
    command = [
        "yolo",
        "detect",
        "predict",
        f"model={remote_model}",
        f"source={remote_source}",
        f"project={remote_project}",
        f"name={run_name}",
        "save=True",
        "exist_ok=True",
    ]
    command_text = " ".join(shlex.quote(str(part)) for part in command)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    sftp = None
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
        sftp = client.open_sftp()
        sftp_mkdirs(sftp, remote_source)
        sftp_upload_tree(sftp, upload_dir, remote_source)
        sftp_upload_file(sftp, model_path, remote_model)
        stdin, stdout, stderr = client.exec_command(
            f"mkdir -p {shlex.quote(remote_project)} && {command_text}",
            timeout=3600,
        )
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        status_code = stdout.channel.recv_exit_status()
        if status_code != 0:
            return {
                "ok": False,
                "error": stderr_text or stdout_text or f"远程预测失败，退出码 {status_code}",
                "command": command_text,
                "outputs": [],
            }
        sftp_download_tree(sftp, remote_run_dir, local_run_dir)
        return {
            "ok": True,
            "error": "",
            "command": f"{resource.get('name') or resource.get('host')} $ {command_text}",
            "outputs": output_items(project, path, local_run_dir, model_name),
        }
    except Exception as error:
        return {
            "ok": False,
            "error": f"远程预测失败：{error}",
            "command": command_text,
            "outputs": [],
        }
    finally:
        if sftp is not None:
            try:
                sftp.close()
            except Exception:
                pass
        try:
            client.exec_command(f"rm -rf {shlex.quote(remote_root)}", timeout=20)
        except Exception:
            pass
        client.close()


def predict_model_items(path: Path, current_project: str):
    items = []
    seen = set()
    for model in train_model_items(load_train_tasks(), current_project):
        run_dir = Path(model.get("run_dir") or "")
        weights_dir = run_dir / "weights"
        model_file = weights_dir / "best.pt"
        if not model_file.is_file():
            model_file = weights_dir / "last.pt"
        if not model_file.is_file():
            continue
        resolved = model_file.resolve()
        if not is_inside(resolved, path.resolve()):
            continue
        relative_path = resolved.relative_to(path.resolve()).as_posix()
        seen.add(relative_path)
        task = model.get("task") or {}
        items.append(
            {
                "name": task.get("name") or run_dir.name,
                "relative_path": relative_path,
            }
        )
    for model in val_model_items(path):
        relative_path = str(model.get("relative_path") or "")
        if relative_path and relative_path not in seen:
            items.append(model)
    return items


def predict_context(
    request: Request,
    current_project: str,
    result: dict | None = None,
    selected_compute: str = "local",
    selected_models: list[str] | None = None,
):
    workspace = workspace_path()
    path = project_path(workspace, current_project)
    return {
        "request": request,
        "workspace": workspace,
        "active_page": "model",
        "model_active": "predict",
        "current_project": current_project,
        "project_name": read_project_name(path) if path else "",
        "models": predict_model_items(path, current_project) if path else [],
        "resources": read_resources(workspace),
        "selected_compute": selected_compute or "local",
        "selected_models": selected_models or [],
        "result": result,
        "mobile_url": f"/model/{current_project}/predict/mobile" if current_project else "",
        "mobile_qr_url": f"/model/{current_project}/predict/mobile/qr.svg" if current_project else "",
        **header_context(request, workspace),
    }


def absolute_url(request: Request, path: str):
    return str(request.base_url).rstrip("/") + path


@router.get("/model/predict")
@router.get("/predict")
def predict(request: Request):
    project = request.query_params.get("project", "")
    if project:
        return RedirectResponse(url=f"/model/{project}/predict", status_code=status.HTTP_303_SEE_OTHER)
    current_project = request.cookies.get("current_project", "")
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context=predict_context(request, current_project),
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/predict")
@router.get("/model/predict/{project}")
@router.get("/predict/{project}")
def predict_with_project(request: Request, project: str):
    if request.url.path.startswith("/model/predict/"):
        return RedirectResponse(url=f"/model/{project}/predict", status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context=predict_context(request, project),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/predict/mobile/qr.svg")
def predict_mobile_qr(request: Request, project: str):
    path = project_path(workspace_path(), project)
    if path is None:
        return Response(status_code=404)
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return Response("qrcode dependency is missing", status_code=500)
    image = qrcode.make(
        absolute_url(request, f"/model/{project}/predict/mobile"),
        image_factory=qrcode.image.svg.SvgPathImage,
        box_size=10,
        border=2,
    )
    buffer = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
    image.save(buffer)
    buffer.seek(0)
    return Response(buffer.read(), media_type="image/svg+xml")


@router.get("/model/{project}/predict/mobile")
def predict_mobile(request: Request, project: str):
    workspace = workspace_path()
    path = project_path(workspace, project)
    if path is None:
        return RedirectResponse(url="/model/predict", status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="predict/mobile.html",
        context={
            "request": request,
            "workspace": workspace,
            "current_project": project,
            "project_name": read_project_name(path) if path else "",
            "models": predict_model_items(path, project),
            "selected_model": request.query_params.get("model") or default_model_value(path, project),
        },
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.post("/model/{project}/predict/mobile/frame")
async def predict_mobile_frame(project: str, request: Request):
    workspace = workspace_path()
    path = project_path(workspace, project)
    if path is None:
        return JSONResponse({"ok": False, "error": "项目不存在"}, status_code=404)
    form = await request.form()
    model_value = str(form.get("model") or default_model_value(path, project)).strip()
    model_path = resolve_model(path, model_value)
    if model_path is None:
        return JSONResponse({"ok": False, "error": "请选择有效模型"}, status_code=400)
    image = form.get("image")
    if not isinstance(image, UploadFile) and not (hasattr(image, "filename") and hasattr(image, "read")):
        return JSONResponse({"ok": False, "error": "缺少图像帧"}, status_code=400)
    content = await image.read()
    if not content:
        return JSONResponse({"ok": False, "error": "图像帧为空"}, status_code=400)
    frame_dir = predict_root(path) / "mobile"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    frame_path.write_bytes(content)
    try:
        boxes = predict_image_boxes(model_path, frame_path)
    except Exception as error:
        return JSONResponse({"ok": False, "error": str(error)}, status_code=500)
    finally:
        try:
            frame_path.unlink()
        except OSError:
            pass
    labels = []
    seen = set()
    for box in boxes:
        label = str(box.get("label") or "")
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return {"ok": True, "boxes": boxes, "labels": labels}


@router.post("/model/{project}/predict")
@router.post("/model/predict/{project}")
@router.post("/predict/{project}")
async def run_predict_with_project(request: Request, project: str):
    workspace = workspace_path()
    path = project_path(workspace, project)
    result = {"ok": False, "error": "项目不存在", "outputs": []}
    compute_target = "local"
    selected_models = []
    if path:
        form = await request.form()
        compute_target = str(form.get("compute_target") or "local")
        selected_models = [str(value or "") for value in form.getlist("models") if str(value or "").strip()]
        if not selected_models:
            single_model = str(form.get("model") or "").strip()
            if single_model:
                selected_models = [single_model]
        files = [
            value
            for key, value in form.multi_items()
            if key == "media" and hasattr(value, "filename") and hasattr(value, "read")
        ]
        model_entries = []
        model_lookup = {str(item.get("relative_path") or ""): item for item in predict_model_items(path, project)}
        for model_value in selected_models:
            model_path = resolve_model(path, model_value)
            if model_path is not None:
                model_entries.append((model_value, model_path, model_lookup.get(model_value, {}).get("name") or model_path.stem))
        if not model_entries:
            result = {"ok": False, "error": "请至少选择一个有效模型", "outputs": []}
        elif not files:
            result = {"ok": False, "error": "请上传图片或视频", "outputs": []}
        else:
            run_name, upload_dir, saved = await save_upload_files(path, files, str(form.get("capture_kind") or ""))
            if not saved:
                result = {"ok": False, "error": "没有可用的图片或视频文件", "outputs": []}
            elif compute_target != "local":
                resource = find_resource(workspace, compute_target)
                if resource is None:
                    result = {"ok": False, "error": "请选择有效算力服务器", "outputs": []}
                else:
                    outputs = []
                    commands = []
                    errors = []
                    for index, (_model_value, model_path, model_name) in enumerate(model_entries, start=1):
                        item_run_name = f"{run_name}_{index:02d}"
                        item_result = run_remote_predict(project, path, resource, model_path, upload_dir, item_run_name, model_name)
                        outputs.extend(item_result.get("outputs") or [])
                        if item_result.get("command"):
                            commands.append(item_result["command"])
                        if not item_result.get("ok"):
                            errors.append(f"{model_name}: {item_result.get('error') or '预测失败'}")
                    result = {"ok": not errors, "error": "\n".join(errors), "command": "\n".join(commands), "outputs": outputs}
                    result["saved"] = [item.name for item in saved]
            else:
                outputs = []
                commands = []
                errors = []
                for index, (_model_value, model_path, model_name) in enumerate(model_entries, start=1):
                    item_run_name = f"{run_name}_{index:02d}"
                    item_result = run_predict(project, path, model_path, upload_dir, item_run_name, model_name)
                    outputs.extend(item_result.get("outputs") or [])
                    if item_result.get("command"):
                        commands.append(item_result["command"])
                    if not item_result.get("ok"):
                        errors.append(f"{model_name}: {item_result.get('error') or '预测失败'}")
                result = {"ok": not errors, "error": "\n".join(errors), "command": "\n".join(commands), "outputs": outputs}
                result["saved"] = [item.name for item in saved]
    response = templates.TemplateResponse(
        request=request,
        name="predict/index.html",
        context=predict_context(request, project, result, compute_target, selected_models),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/predict/files/{file_path:path}")
@router.get("/model/predict/{project}/files/{file_path:path}")
@router.get("/predict/{project}/files/{file_path:path}")
def predict_file(project: str, file_path: str):
    workspace = workspace_path()
    path = project_path(workspace, project)
    if path is None:
        return RedirectResponse(url="/model/predict", status_code=status.HTTP_303_SEE_OTHER)
    file = (path / file_path).resolve()
    allowed_roots = ((path / "predict-runs").resolve(), (path / "predict" / "uploads").resolve())
    if not file.is_file() or not any(file == root or root in file.parents for root in allowed_roots):
        return RedirectResponse(url=f"/model/{project}/predict", status_code=status.HTTP_303_SEE_OTHER)
    return FileResponse(file)
