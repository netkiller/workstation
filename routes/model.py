import math
from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from routes.edition import is_community_edition
from routes.project import (
    PROJECT_TASK_CLASSIFY,
    header_context,
    project_dir as project_root_dir,
    project_task_type,
    read_project_meta,
    read_project_registry,
)
from routes.train import load_tasks as load_train_tasks, model_items as run_model_items, read_metric_rows
from routes.val import dataset_items, project_path, read_project_name, workspace_path


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


def project_task_path(project: str):
    path = project_root_dir(workspace_path(), project)
    if path is None or not path.is_dir():
        return "detect"
    meta = read_project_meta(path, read_project_registry(path.parent))
    return "classify" if project_task_type(str(meta.get("task_type") or "")) == PROJECT_TASK_CLASSIFY else "detect"


def _metric_number(value):
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def _format_radar_value(value, digits=3):
    if value is None:
        return "0.000" if digits else "0"
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def _radar_context(
    metric_labels,
    raw_series,
    normalize_by_axis=True,
    radius=60,
    label_scale=1.14,
    center_y=70,
    side_label_offset=0,
    view_box="0 0 150 132",
):
    colors = ("#1667c7", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#dc2626", "#0f766e", "#c2410c")
    max_values = {
        label: max((series["values"].get(label, 0) for series in raw_series), default=0)
        for label in metric_labels
    }

    center_x = 75

    def point(index, scale=1):
        angle = math.radians(-90 + index * 360 / len(metric_labels))
        return {
            "x": center_x + math.cos(angle) * radius * scale,
            "y": center_y + math.sin(angle) * radius * scale,
        }

    def points_string(points):
        return " ".join(f"{item['x']:.1f},{item['y']:.1f}" for item in points)

    axes = []
    for index, label in enumerate(metric_labels):
        end = point(index)
        label_point = point(index, label_scale)
        if side_label_offset and label == "mAP50":
            label_point["x"] += side_label_offset
        elif side_label_offset and label == "Recall":
            label_point["x"] -= side_label_offset
        axes.append({"label": label, **end, "label_x": label_point["x"], "label_y": label_point["y"]})

    series_items = []
    for index, series in enumerate(raw_series):
        normalized = []
        for label in metric_labels:
            value = series["values"][label]
            if normalize_by_axis:
                max_value = max_values.get(label, 0)
                normalized.append(min(value / max_value, 1) if max_value else 0)
            else:
                normalized.append(min(max(value, 0), 1))
        value_points = [point(point_index, normalized_value) for point_index, normalized_value in enumerate(normalized)]
        polygon = points_string(value_points)
        series_items.append(
            {
                "name": series["name"],
                "color": colors[index % len(colors)],
                "polygon": polygon,
                "closed_polygon": f"{polygon} {value_points[0]['x']:.1f},{value_points[0]['y']:.1f}" if value_points else "",
                "points": value_points,
                "metrics": [
                    {
                        "label": label,
                        "value": _format_radar_value(series["values"][label], digits=0 if label == "Epochs" else 3),
                    }
                    for label in metric_labels
                ],
            }
        )

    return {
        "axes": axes,
        "grid": [points_string([point(index, scale) for index in range(len(metric_labels))]) for scale in (0.2, 0.4, 0.6, 0.8, 1)],
        "series": series_items,
        "center_x": center_x,
        "center_y": center_y,
        "view_box": view_box,
    }


def model_radar_context(models):
    metric_labels = ("Epochs", "mAP50", "mAP50-95", "Precision", "Recall")
    raw_series = []
    for model in models:
        lookup = {
            metric.get("label"): _metric_number(metric.get("value"))
            for metric in (model.get("metrics") or [])
        }
        if not any(lookup.get(label) is not None for label in metric_labels):
            continue
        task = model.get("task") or {}
        raw_series.append(
            {
                "name": task.get("name") or "未命名模型",
                "values": {label: lookup.get(label) or 0 for label in metric_labels},
            }
        )
    return _radar_context(
        metric_labels,
        raw_series,
        radius=64,
        label_scale=1.06,
        center_y=70,
        side_label_offset=8,
        view_box="-10 0 170 132",
    )


def csv_radar_context(models, metric_labels):
    raw_series = []
    for model in models:
        if not model.get("run_dir"):
            continue
        run_dir = Path(model.get("run_dir"))
        rows = read_metric_rows(run_dir / "results.csv")
        if not rows:
            continue
        row = rows[-1]
        if not any(row.get(label) not in (None, "") for label in metric_labels):
            continue
        values = {label: _metric_number(row.get(label)) or 0 for label in metric_labels}
        task = model.get("task") or {}
        raw_series.append({"name": task.get("name") or "未命名模型", "values": values})
    return _radar_context(metric_labels, raw_series, radius=52, label_scale=1.12, center_y=70)


def model_item_task(model):
    task = model.get("task") or {}
    task_type = str(task.get("task_type") or "")
    if task_type:
        return project_task_type(task_type)
    return project_task_path(str(task.get("project") or ""))


def filter_models_by_task(models, task: str):
    task = "classify" if task == "classify" else "detect"
    return [model for model in models if model_item_task(model) == task]


def model_context(request: Request, current_project: str, task: str = "detect"):
    workspace = workspace_path()
    path = project_path(workspace, current_project)
    task = "classify" if task == "classify" else "detect"
    models = filter_models_by_task(run_model_items(load_train_tasks(), current_project), task) if path else []
    return {
        "request": request,
        "workspace": workspace,
        "active_page": "model",
        "model_active": "overview",
        "current_project": current_project,
        "model_task": task,
        "project_name": read_project_name(path) if path else "",
        "models": models,
        "community_edition": is_community_edition(),
        "model_radar": model_radar_context(models),
        "loss_radar": csv_radar_context(
            models,
            ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"),
        ),
        "lr_radar": csv_radar_context(
            models,
            ("lr/pg0", "lr/pg1", "lr/pg2", "lr/pg3", "lr/pg4", "lr/pg5", "lr/pg6", "lr/pg7"),
        ),
        "datasets": dataset_items(path) if path else [],
        **header_context(request, workspace),
    }


def analysis_context(request: Request, current_project: str):
    workspace = workspace_path()
    path = project_path(workspace, current_project)
    return {
        "request": request,
        "workspace": workspace,
        "active_page": "model",
        "model_active": "analysis",
        "current_project": current_project,
        "project_name": read_project_name(path) if path else "",
        **header_context(request, workspace),
    }


@router.get("/model")
def model_index(request: Request):
    project = request.query_params.get("project", "")
    if project:
        task = project_task_path(project)
        return RedirectResponse(url=f"/model/{project}/{task}", status_code=status.HTTP_303_SEE_OTHER)
    current_project = request.cookies.get("current_project", "")
    if current_project:
        task = project_task_path(current_project)
        return RedirectResponse(url=f"/model/{current_project}/{task}", status_code=status.HTTP_303_SEE_OTHER)
    response = templates.TemplateResponse(
        request=request,
        name="model/index.html",
        context=model_context(request, current_project),
    )
    if current_project:
        response.set_cookie("current_project", current_project, httponly=True, samesite="lax")
    return response


@router.get("/model/analysis")
def model_analysis_index(request: Request):
    project = request.query_params.get("project", "")
    current_project = project or request.cookies.get("current_project", "")
    if current_project:
        return RedirectResponse(url=f"/model/{current_project}/analysis", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(
        request=request,
        name="model/analysis.html",
        context=analysis_context(request, ""),
    )


@router.get("/model/{project}/analysis")
@router.get("/model/{project}/{task}/analysis")
def model_analysis_project(request: Request, project: str, task: str = "detect"):
    response = templates.TemplateResponse(
        request=request,
        name="model/analysis.html",
        context=analysis_context(request, project),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}/{task}")
def model_project_task(request: Request, project: str, task: str):
    if task not in {"detect", "classify"}:
        return JSONResponse({"ok": False, "error": "任务类型不存在"}, status_code=404)
    template_name = "model/classify.html" if task == "classify" else "model/index.html"
    response = templates.TemplateResponse(
        request=request,
        name=template_name,
        context=model_context(request, project, task),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response


@router.get("/model/{project}")
def model_project(request: Request, project: str):
    task = project_task_path(project)
    response = templates.TemplateResponse(
        request=request,
        name="model/classify.html" if task == "classify" else "model/index.html",
        context=model_context(request, project, task),
    )
    response.set_cookie("current_project", project, httponly=True, samesite="lax")
    return response
