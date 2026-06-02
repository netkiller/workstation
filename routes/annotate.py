import os
import asyncio
import json
import html
import re
import traceback
from pathlib import Path
from urllib.parse import quote

from fastapi import Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from routes.project import (
    CLASSIFY_DIR,
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
        if path == self.workspace and path.name in {"annotate", CLASSIFY_DIR}:
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


def annotate_base_html(team_mode: bool, template_section: str = "annotate"):
    framework = (TEMPLATES_DIR / "framework.html").read_text(encoding="utf-8")
    annotate = (TEMPLATES_DIR / template_section / "index.html").read_text(encoding="utf-8")
    footer = (TEMPLATES_DIR / "partials" / "footer.html").read_text(encoding="utf-8")
    shortcut_match = re.search(
        r"\{# ANNOTATE_SHORTCUTS_START #\}(.*?)\{# ANNOTATE_SHORTCUTS_END #\}",
        footer,
        flags=re.S,
    )
    shortcut_popover = shortcut_match.group(1).strip() if shortcut_match else ""
    body_classes = [f"{template_section}-mode"]
    if team_mode:
        body_classes.extend(["team-mode", "username-required"])
    body_class = " ".join(body_classes)
    return (
        framework
        .replace("__ANNOTATE_CONTENT__", annotate)
        .replace("__FOOTER_CONTENT__", shortcut_popover)
        .replace("__BODY_CLASS__", body_class)
        .replace("__TEAM_MODE__", "true" if team_mode else "false")
    )


def workstation_html(
    workstation: Workstation,
    active_mode: str = "annotate",
    project: str = "",
    username: str = "",
    route_prefix: str = "annotate",
    template_section: str = "annotate",
    mode_label: str = "标注",
    mode_icon: str = "▧",
):
    html = annotate_base_html(workstation.team_mode, template_section)
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
        '<span class="nav-icon header-icon">▤</span><span>项目</span></button>'
    )
    resources_url = f"/resources/{quote(project, safe='')}" if project else "/resources"
    resources_button = (
        '<button id="resourcesButton" class="header-button" title="算力" '
        f'onclick="location.href=\'{resources_url}\'">'
        '<span class="nav-icon header-icon">▥</span><span>算力</span></button>'
    )
    task_path = "classify" if route_prefix == "classify" else "detect"
    dataset_url = f"/dataset/{quote(project, safe='')}/{task_path}" if project else "/dataset"
    model_url = f"/model/{quote(project, safe='')}/{task_path}" if project else "/model"
    test_url = f"/test/{quote(project, safe='')}/{task_path}" if project else "/test"
    test_button = (
        '<button id="testButton" class="header-button" title="测试" '
        f'onclick="location.href=\'{test_url}\'">'
        '<span class="nav-icon header-icon">◉</span><span>测试</span></button>'
    )
    team_button = (
        '<button id="teamButton" class="header-button" title="团队" '
        f'onclick="location.href=\'{team_url}\'">'
        '<span class="nav-icon header-icon">◎</span><span>团队</span></button>'
        if team_mode_enabled()
        else ""
    )
    user_header = ""
    html = (
        html.replace('"/api/', f'"/{route_prefix}/api/')
        .replace("'/api/", f"'/{route_prefix}/api/")
        .replace("`/api/", f"`/{route_prefix}/api/")
        .replace('"/media', f'"/{route_prefix}/media')
        .replace("'/media", f"'/{route_prefix}/media")
        .replace("`/media", f"`/{route_prefix}/media")
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
            '</button>\n    </nav>\n    <div class="header-actions">',
            f'</button>{test_button}\n    </nav>\n    <div class="header-actions">',
            1,
        )
        .replace(
            'id="annotateModeButton" class="header-button active"',
            f'id="annotateModeButton" class="header-button" onclick="location.href=\'/{route_prefix}/{quote(project, safe="")}\'"',
        )
        .replace(
            'id="datasetButton" class="header-button"',
            f'id="datasetButton" class="header-button" onclick="location.href=\'{dataset_url}\'"',
        )
        .replace(
            'id="modelButton" class="header-button"',
            f'id="modelButton" class="header-button" onclick="location.href=\'{model_url}\'"',
        )
        .replace(
            'datasetButton.addEventListener("click", showEnterpriseNotice);',
            f'datasetButton.addEventListener("click", () => {{ location.href = "{dataset_url}"; }});',
        )
        .replace(
            'modelButton.addEventListener("click", showEnterpriseNotice);',
            f'modelButton.addEventListener("click", () => {{ location.href = "{model_url}"; }});',
        )
    )
    active_button = {
        "annotate": "annotateModeButton",
        "classify": "annotateModeButton",
        "dataset": "datasetButton",
        "model": "modelButton",
    }.get(active_mode)
    if active_button:
        html = html.replace(
            f'id="{active_button}" class="header-button"',
            f'id="{active_button}" class="header-button active"',
        )
    if active_mode == "classify":
        for button_id in ("autoAnnotate", "maskAnnotation", "editModeToggle"):
            html = html.replace(
                f'id="{button_id}" class="header-button"',
                f'id="{button_id}" hidden class="header-button"',
                1,
            )
    if username:
        html = re.sub(
            r'<body class="([^"]*)\busername-required\b\s*([^"]*)"',
            lambda match: f'<body class="{(" ".join((match.group(1) + match.group(2)).split()))}"',
            html,
            count=1,
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
    api_project_script = """
        (() => {
          const project = window.yoloutilsProject || "";
          if (!project) return;
          const nativeFetch = window.fetch.bind(window);
          window.fetch = (input, init) => {
            const url = typeof input === "string" ? input : input?.url;
            if (typeof url !== "string" || !url.startsWith("/__ROUTE_PREFIX__/api/")) {
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
        """.replace("__ROUTE_PREFIX__", route_prefix)
    user_script = (
        "<script>"
        f"window.yoloutilsUsername = {json.dumps(username, ensure_ascii=False)};"
        f"window.yoloutilsProject = {json.dumps(project, ensure_ascii=False)};"
        f"window.yoloutilsOnlineUsers = {json.dumps(online_users, ensure_ascii=False)};"
        f"{api_project_script}"
        "window.yoloutilsUsernameReady = Promise.resolve(window.yoloutilsUsername);"
        "try { localStorage.setItem('yoloutils-workstation-username', window.yoloutilsUsername); } catch (_) {}"
        "document.addEventListener('DOMContentLoaded', () => document.body.classList.remove('username-required'));"
        "</script>"
    )
    html = html.replace(
        '<span class="nav-icon header-icon">▧</span><span>标注</span>',
        f'<span class="nav-icon header-icon">{html_escape(mode_icon)}</span><span>{html_escape(mode_label)}</span>',
        1,
    )
    html = html.replace('<span class="header-icon">▧</span><span>标注</span>', f'<span class="header-icon">{html_escape(mode_icon)}</span><span>{html_escape(mode_label)}</span>')
    return html.replace("</head>", f"{user_script}</head>", 1)


def html_escape(value: str):
    return html.escape(value or "", quote=True)


def create_workstation():
    workstation = SiteWorkstation()
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


def apply_project_workspace(workstation: Workstation, project: str, workspace_getter=project_annotate_workspace):
    images_dir = workspace_getter(project)
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


def register_annotate_routes(
    app,
    workstation: Workstation,
    *,
    route_prefix: str = "annotate",
    template_section: str = "annotate",
    workspace_getter=project_annotate_workspace,
    active_mode: str = "annotate",
    mode_label: str = "标注",
    mode_icon: str = "▧",
    error_label: str = "Annotate",
):
    app.state.project_workspace_lock = asyncio.Lock()

    @app.middleware("http")
    async def project_workspace_middleware(request: Request, call_next):
        project = request.query_params.get("project") or request.path_params.get("project") or request.cookies.get("current_project", "")
        async with app.state.project_workspace_lock:
            if project:
                apply_project_workspace(workstation, project, workspace_getter)
            else:
                apply_project_workspace(workstation, "", workspace_getter)
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
            apply_project_workspace(workstation, project, workspace_getter)
            response = HTMLResponse(workstation_html(
                workstation,
                active_mode,
                project,
                username,
                route_prefix=route_prefix,
                template_section=template_section,
                mode_label=mode_label,
                mode_icon=mode_icon,
            ))
            if project:
                response.set_cookie("current_project", project, httponly=True, samesite="lax")
            return response
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                f"{error_label} page error. See .yoloutils-annotate-error.log",
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
            apply_project_workspace(workstation, project, workspace_getter)
            response = HTMLResponse(workstation_html(
                workstation,
                active_mode,
                project,
                username,
                route_prefix=route_prefix,
                template_section=template_section,
                mode_label=mode_label,
                mode_icon=mode_icon,
            ))
            response.set_cookie("current_project", project, httponly=True, samesite="lax")
            return response
        except Exception as error:
            write_error_log(workstation, error)
            return PlainTextResponse(
                f"{error_label} page error. See .yoloutils-annotate-error.log",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    return app
