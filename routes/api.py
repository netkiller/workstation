import time

from fastapi import APIRouter, HTTPException, Query, Request


def create_workstation_api_router(workstation):
    router = APIRouter(prefix="/api")

    @router.get("/config")
    def config():
        return {"team_mode": workstation.team_mode, "share_url": workstation._share_url() if workstation.team_mode else ""}

    @router.get("/tree")
    def tree(directory: str = Query(default="")):
        path = workstation._safe_path(directory)
        if not path.is_dir():
            raise HTTPException(status_code=404, detail="directory not found")
        return workstation._directory_tree(path)

    @router.get("/files")
    def files(directory: str = Query(default="")):
        return {"directory": directory, "files": workstation._list_files(directory)}

    @router.get("/models")
    def models():
        return {"models": workstation._model_files()}

    @router.post("/models/upload")
    async def upload_models(request: Request):
        saved = await workstation._save_model_uploads(request)
        if not saved:
            raise HTTPException(status_code=400, detail="请选择有效模型文件")
        return {"ok": True, "saved": len(saved), "models": workstation._model_files()}

    @router.post("/models/delete")
    async def delete_model(request: Request):
        payload = await request.json()
        model_file = workstation._safe_model_path(str(payload.get("path", "")))
        model_key = str(model_file)
        try:
            model_file.unlink()
        except OSError as error:
            raise HTTPException(status_code=500, detail=f"删除模型失败: {error}") from error
        workstation.model_cache.pop(model_key, None)
        return {"ok": True, "models": workstation._model_files()}

    @router.get("/classes")
    def classes():
        return {
            "classes_file": workstation.class_groups[0]["classes_file"] if workstation.class_groups else None,
            "classes": workstation.classes,
            "content": workstation._classes_text(),
            "class_groups": [
                {
                    "classes_file": group["classes_file"],
                    "classes": group["classes"],
                }
                for group in workstation.class_groups
            ],
        }

    @router.post("/classes")
    async def save_classes(request: Request):
        payload = await request.json()
        target = workstation._save_classes_text(str(payload.get("content", "")))
        return {
            "ok": True,
            "classes_file": workstation._relative(target) if workstation._is_inside_workspace(target) else str(target),
            "classes": workstation.classes,
            "content": workstation._classes_text(),
        }

    @router.post("/classes/action")
    async def classes_action(request: Request):
        payload = await request.json()
        target = payload.get("target_index")
        try:
            index = int(payload.get("index", -1))
            target_index = int(target) if target not in (None, "") else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="标签索引无效")
        return workstation._rewrite_label_indices(
            index,
            str(payload.get("action", "")).strip(),
            target_index,
            str(payload.get("directory", "")).strip(),
        )

    @router.get("/statistics")
    def statistics():
        return workstation._statistics()

    @router.post("/presence")
    async def presence(request: Request):
        payload = await request.json()
        client_id = str(payload.get("client_id", "")).strip()
        username = str(payload.get("username", "")).strip() or "独立用户"
        if not client_id:
            raise HTTPException(status_code=400, detail="client_id required")
        workstation._online_count()
        username_key = username.casefold()
        for current_client_id, info in workstation.presence.items():
            if current_client_id == client_id:
                continue
            if str(info.get("username", "")).casefold() == username_key:
                raise HTTPException(status_code=409, detail="用户名已被使用")
        workstation.presence[client_id] = {
            "username": username,
            "seen_at": time.time(),
        }
        return {"online": workstation._online_count(), "users": workstation._online_users()}

    @router.post("/presence/leave")
    async def leave_presence(request: Request):
        payload = await request.json()
        return workstation._leave_presence(str(payload.get("client_id", "")).strip())

    @router.post("/lock")
    async def lock(request: Request):
        payload = await request.json()
        return workstation._acquire_lock(
            payload.get("path", ""),
            str(payload.get("client_id", "")).strip(),
            str(payload.get("username", "")).strip(),
        )

    @router.post("/lock/release")
    async def release_lock(request: Request):
        payload = await request.json()
        return workstation._release_lock(
            str(payload.get("path", "")).strip(),
            str(payload.get("client_id", "")).strip(),
        )

    @router.get("/logs")
    def logs(lines: int = Query(default=200, ge=1, le=1000)):
        return workstation._read_logs(lines)

    @router.get("/annotation")
    def annotation(path: str):
        return workstation._read_annotation(path)

    @router.post("/annotation")
    async def save_annotation(request: Request):
        payload = await request.json()
        return workstation._write_annotation(
            payload.get("path", ""),
            payload.get("boxes", []),
            str(payload.get("client_id", "")).strip(),
            str(payload.get("username", "")).strip(),
        )

    @router.post("/auto-annotate")
    async def auto_annotate(request: Request):
        payload = await request.json()
        return workstation._predict_boxes(
            payload.get("path", ""),
            str(payload.get("model", "")).strip(),
        )

    @router.post("/file/delete")
    async def delete_file(request: Request):
        payload = await request.json()
        return workstation._delete_image_file(
            payload.get("path", ""),
            str(payload.get("client_id", "")).strip(),
            str(payload.get("username", "")).strip(),
        )

    @router.post("/directory/action")
    async def directory_action(request: Request):
        payload = await request.json()
        return workstation._directory_operation(
            str(payload.get("path", "")).strip(),
            str(payload.get("action", "")).strip(),
            str(payload.get("name", "")).strip(),
        )

    @router.post("/file/action")
    async def file_action(request: Request):
        payload = await request.json()
        return workstation._file_operation(
            str(payload.get("path", "")).strip(),
            str(payload.get("action", "")).strip(),
            str(payload.get("name", "")).strip(),
        )

    @router.get("/exif")
    def exif(path: str):
        return workstation._read_exif(path)

    return router
