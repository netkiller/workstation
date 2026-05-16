import os
import asyncio
import json
import html
import traceback
from pathlib import Path
from urllib.parse import quote

from fastapi import Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from routes.project import (
    current_username as project_current_username,
    project_annotate_workspace,
    project_display_name,
    project_icon,
    project_root_workspace,
    read_project_meta,
    read_project_registry,
    read_online_users,
    team_mode_enabled,
    user_color,
    user_items,
    workspace_path,
    write_user_project,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"

from workstation import Workstation


class SiteWorkstation(Workstation):
    def _directory_tree(self, path: Path, include_children: bool = True):
        tree = super()._directory_tree(path, include_children=include_children)
        if path == self.workspace and path.name == "annotate":
            tree["name"] = getattr(self, "root_label", "") or "根目录"
        return tree


def site_workspace():
    return workspace_path()


def write_error_log(workstation: Workstation, error: Exception):
    workspace = workstation.workspace if workstation.workspace else PROJECT_ROOT
    log_file = workspace / ".yoloutils-annotate-error.log"
    try:
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        fallback = PROJECT_ROOT / ".yoloutils-annotate-error.log"
        fallback.write_text(traceback.format_exc(), encoding="utf-8")


def annotate_base_html(team_mode: bool):
    framework = (TEMPLATES_DIR / "framework.html").read_text(encoding="utf-8")
    annotate = (TEMPLATES_DIR / "annotate" / "index.html").read_text(encoding="utf-8")
    body_class = "team-mode username-required" if team_mode else ""
    return (
        framework
        .replace("__ANNOTATE_CONTENT__", annotate)
        .replace("__BODY_CLASS__", body_class)
        .replace("__TEAM_MODE__", "true" if team_mode else "false")
    )


def workstation_html(workstation: Workstation, active_mode: str = "annotate", project: str = "", username: str = ""):
    html = annotate_base_html(workstation.team_mode)
    escaped_username = html_escape(username)
    workspace = site_workspace()
    online_users = user_items(read_online_users(workspace))
    project_url = f"/project/{quote(project, safe='')}" if project else "/project"
    project_query = f"?project={quote(project, safe='')}" if project else ""
    team_url = f"/team/{quote(project, safe='')}" if project else "/team"
    project_root = project_root_workspace(project) if project else None
    project_meta = read_project_meta(project_root, read_project_registry(workspace)) if project_root and project_root.is_dir() else None
    header_project_name = project_meta["name"] if project_meta else ""
    header_project_icon = project_meta["icon"] if project_meta else project_icon("")
    close_project_button = ""
    project_button = (
        '<button id="projectButton" class="header-button" '
        f'title="项目" onclick="location.href=\'{project_url}\'">'
        '<span class="header-icon">▤</span><span>项目</span></button>'
    )
    resources_url = f"/resources/{quote(project, safe='')}" if project else "/resources"
    resources_button = (
        '<button id="resourcesButton" class="header-button" title="算力" '
        f'onclick="location.href=\'{resources_url}\'">'
        '<span class="header-icon">▥</span><span>算力</span></button>'
    )
    test_url = f"/test/{quote(project, safe='')}" if project else "/test"
    test_button = (
        '<button id="testButton" class="header-button" title="测试" '
        f'onclick="location.href=\'{test_url}\'">'
        '<span class="header-icon">◉</span><span>测试</span></button>'
    )
    team_button = (
        '<button id="teamButton" class="header-button" title="团队" '
        f'onclick="location.href=\'{team_url}\'">'
        '<span class="header-icon">◌</span><span>团队</span></button>'
        if team_mode_enabled()
        else ""
    )
    user_header = (
        ""
        f'<span class="enterprise-link user-avatar-link" style="background:{user_color(username)}">'
        f'{html_escape(username[:1])}</span>'
        f'<span class="enterprise-link username-link">{escaped_username}</span>'
        '<form method="post" action="/team/logout" style="margin:0"><button class="enterprise-link" type="submit">注销</button></form>'
        if team_mode_enabled()
        else ""
    )
    html = (
        html.replace('"/api/', '"/annotate/api/')
        .replace("'/api/", "'/annotate/api/")
        .replace("`/api/", "`/annotate/api/")
        .replace('"/media', '"/annotate/media')
        .replace("'/media", "'/annotate/media")
        .replace("`/media", "`/annotate/media")
        .replace(
            '<a class="brand-link" href="https://www.netkiller.cn" target="_blank" rel="noopener noreferrer">Yolo Workstation</a>',
            (
                f'<a class="brand-link" href="{project_url}" title="{html_escape(header_project_name)}">'
                f"{html_escape(header_project_name)}</a>{user_header}{close_project_button}"
                if project and header_project_name
                else f"{user_header}{close_project_button}"
            ),
            1,
        )
        .replace(
            '<a class="header-home-link" href="https://saas.netkiller.cn" target="_blank" rel="noopener noreferrer" title="Home" aria-label="Home">\n        <svg class="header-home-icon" viewBox="0 0 24 24" aria-hidden="true">\n          <path d="M3 11.5 12 4l9 7.5"></path>\n          <path d="M5.5 10.5V20h13v-9.5"></path>\n          <path d="M9.5 20v-6h5v6"></path>\n        </svg>\n      </a>',
            (
                f'<a class="header-home-link header-project-icon-link" href="{project_url}" '
                f'title="{html_escape(header_project_name)}" aria-label="{html_escape(header_project_name)}">'
                f'<span class="header-project-icon">{html_escape(header_project_icon)}</span></a>'
                if project and header_project_name
                else '<a class="header-home-link" href="https://saas.netkiller.cn" target="_blank" rel="noopener noreferrer" title="Home" aria-label="Home">\n        <svg class="header-home-icon" viewBox="0 0 24 24" aria-hidden="true">\n          <path d="M3 11.5 12 4l9 7.5"></path>\n          <path d="M5.5 10.5V20h13v-9.5"></path>\n          <path d="M9.5 20v-6h5v6"></path>\n        </svg>\n      </a>'
            ),
            1,
        )
        .replace(
            "main { height: calc(100vh - 88px);",
            "main { height: calc(100vh - 84px);",
            1,
        )
        .replace(
            "body.console-open main { height: calc(100vh - 88px - var(--console-height, 160px) - 4px);",
            "body.console-open main { height: calc(100vh - 84px - var(--console-height, 160px) - 4px);",
            1,
        )
        .replace(
            '<button id="annotateModeButton"',
            f'{team_button}{project_button}{resources_button}<button id="annotateModeButton"',
            1,
        )
        .replace(
            '</button>\n    </div>\n    <div class="header-actions">',
            f'</button>{test_button}\n    </div>\n    <div class="header-actions">',
            1,
        )
        .replace(
            'id="annotateModeButton" class="header-button active"',
            f'id="annotateModeButton" class="header-button" onclick="location.href=\'/annotate/{quote(project, safe="")}\'"',
        )
        .replace(
            'id="datasetButton" class="header-button"',
            f'id="datasetButton" class="header-button" onclick="location.href=\'/dataset/{quote(project, safe="")}\'"',
        )
        .replace(
            'id="modelButton" class="header-button"',
            f'id="modelButton" class="header-button" onclick="location.href=\'/model/{quote(project, safe="")}\'"',
        )
        .replace(
            'datasetButton.addEventListener("click", showEnterpriseNotice);',
            f'datasetButton.addEventListener("click", () => {{ location.href = "/dataset/{quote(project, safe="")}"; }});',
        )
        .replace(
            'modelButton.addEventListener("click", showEnterpriseNotice);',
            f'modelButton.addEventListener("click", () => {{ location.href = "/model/{quote(project, safe="")}"; }});',
        )
    )
    active_button = {
        "annotate": "annotateModeButton",
        "dataset": "datasetButton",
        "model": "modelButton",
    }.get(active_mode)
    if active_button:
        html = html.replace(
            f'id="{active_button}" class="header-button"',
            f'id="{active_button}" class="header-button active"',
        )
    if username:
        html = html.replace(
            '<body class="team-mode username-required" data-team-mode="true">',
            '<body class="team-mode" data-team-mode="true">',
            1,
        )
        html = html.replace(
            'window.yoloutilsUsername = "";',
            f'window.yoloutilsUsername = {json.dumps(username, ensure_ascii=False)};',
            1,
        )
        html = html.replace(
            "window.yoloutilsUsernameReady = new Promise(resolve => {\n      window.yoloutilsResolveUsername = resolve;\n    });",
            "window.yoloutilsUsernameReady = Promise.resolve(window.yoloutilsUsername);\n    window.yoloutilsResolveUsername = () => {};",
            1,
        )
    if not team_mode_enabled():
        html = html.replace(
            '<button id="shareButton" class="header-button" title="分享当前页面或当前位置"><span class="header-icon">⇪</span><span>分享</span></button>',
            "",
            1,
        )
    user_script = (
        "<script>"
        f"window.yoloutilsUsername = {json.dumps(username, ensure_ascii=False)};"
        f"window.yoloutilsProject = {json.dumps(project, ensure_ascii=False)};"
        f"window.yoloutilsOnlineUsers = {json.dumps(online_users, ensure_ascii=False)};"
        """
        (() => {
          const project = window.yoloutilsProject || "";
          if (!project) return;
          const nativeFetch = window.fetch.bind(window);
          window.fetch = (input, init) => {
            const url = typeof input === "string" ? input : input?.url;
            if (typeof url !== "string" || !url.startsWith("/annotate/api/")) {
              return nativeFetch(input, init);
            }
            const nextUrl = new URL(url, window.location.origin);
            if (!nextUrl.searchParams.has("project")) {
              nextUrl.searchParams.set("project", project);
            }
            if (typeof input === "string") {
              return nativeFetch(`${nextUrl.pathname}${nextUrl.search}`, init);
            }
            return nativeFetch(new Request(`${nextUrl.pathname}${nextUrl.search}`, input), init);
          };
        })();
        """
        "window.yoloutilsUsernameReady = Promise.resolve(window.yoloutilsUsername);"
        "try { localStorage.setItem('yoloutils-workstation-username', window.yoloutilsUsername); } catch (_) {}"
        "document.addEventListener('DOMContentLoaded', () => document.body.classList.remove('username-required'));"
        "</script>"
    )
    return html.replace("</head>", f"{user_script}</head>", 1)


def html_escape(value: str):
    return html.escape(value or "", quote=True)


def create_workstation():
    workstation = SiteWorkstation()
    workstation.host = os.environ.get("YOLOUTILS_HOST", workstation.host)
    try:
        workstation.port = int(os.environ.get("YOLOUTILS_PORT", workstation.port))
    except (TypeError, ValueError):
        pass
    dataset = os.environ.get("YOLOUTILS_DATASET")
    run = os.environ.get("YOLOUTILS_RUN")

    workstation.workspace = site_workspace()
    workstation.model_root = workstation.workspace
    workstation.log_root = workstation.workspace
    workstation.dataset = Path(dataset).expanduser().resolve() if dataset else None
    workstation.run = Path(run).expanduser().resolve() if run else None
    workstation.requested_classes_file = os.environ.get("YOLOUTILS_CLASSES") or None
    workstation.open_browser = False
    workstation.team_mode = os.environ.get("YOLOUTILS_TEAM", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    workstation.mdns = workstation._normalize_mdns(
        os.environ.get("YOLOUTILS_MDNS", "netkiller.local")
    )
    workstation.root_label = "根目录"
    workstation.class_groups = workstation._load_class_groups()
    workstation.classes_file = (
        workstation.class_groups[0]["path"] if workstation.class_groups else None
    )
    workstation.classes = (
        workstation.class_groups[0]["classes"] if workstation.class_groups else []
    )
    return workstation


def apply_project_workspace(workstation: Workstation, project: str):
    images_dir = project_annotate_workspace(project)
    workstation.workspace = images_dir if images_dir is not None else site_workspace()
    project_root = project_root_workspace(project) if images_dir is not None else None
    workstation.model_root = project_root if project_root is not None else workstation.workspace
    workstation.log_root = project_root if project_root is not None else workstation.workspace
    if project_root is not None and images_dir is not None:
        new_log = project_root / ".project.log"
        old_logs = [
            images_dir / ".yoloutils-workstation.log",
            project_root / ".yoloutils-workstation.log",
            project_root / ".yoloutils-upload.log",
        ]
        for old_log in old_logs:
            if not old_log.is_file() or old_log == new_log:
                continue
            try:
                with open(new_log, "a", encoding="utf-8") as target:
                    if new_log.stat().st_size:
                        target.write("\n")
                    target.write(f"[migrated] {old_log.name}\n")
                    target.write(old_log.read_text(encoding="utf-8", errors="replace"))
                old_log.unlink()
            except OSError:
                pass
    workstation.root_label = project_display_name(project) if images_dir is not None else "根目录"
    workstation.class_groups = workstation._load_class_groups()
    workstation.classes_file = (
        workstation.class_groups[0]["path"] if workstation.class_groups else None
    )
    workstation.classes = (
        workstation.class_groups[0]["classes"] if workstation.class_groups else []
    )


def create_annotate_app():
    workstation = create_workstation()
    app = workstation._create_app()
    app.state.project_workspace_lock = asyncio.Lock()

    @app.middleware("http")
    async def project_workspace_middleware(request: Request, call_next):
        project = request.query_params.get("project") or request.path_params.get("project") or request.cookies.get("current_project", "")
        async with app.state.project_workspace_lock:
            if project:
                apply_project_workspace(workstation, project)
            else:
                apply_project_workspace(workstation, "")
            response = await call_next(request)
        if project:
            response.set_cookie("current_project", project, httponly=True, samesite="lax")
        return response

    @app.get("/")
    def index(request: Request):
        try:
            workspace = site_workspace()
            username = project_current_username(request, workspace)
            if team_mode_enabled() and not username:
                return RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
            project = request.query_params.get("project") or request.cookies.get("current_project", "")
            if team_mode_enabled():
                write_user_project(workspace, username, project)
            apply_project_workspace(workstation, project)
            response = HTMLResponse(workstation_html(workstation, "annotate", project, username))
            if project:
                response.set_cookie("current_project", project, httponly=True, samesite="lax")
            return response
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                "Annotate page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @app.get("/{project}")
    def index_project(request: Request, project: str):
        try:
            workspace = site_workspace()
            username = project_current_username(request, workspace)
            if team_mode_enabled() and not username:
                return RedirectResponse(url="/team", status_code=status.HTTP_303_SEE_OTHER)
            if team_mode_enabled():
                write_user_project(workspace, username, project)
            apply_project_workspace(workstation, project)
            response = HTMLResponse(workstation_html(workstation, "annotate", project, username))
            response.set_cookie("current_project", project, httponly=True, samesite="lax")
            return response
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                "Annotate page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app
