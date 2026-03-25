from typing import Optional

from fastapi import Request

from open_webui.config import OPENWEBUI_APP_MODE


def get_app_mode(request: Optional[Request] = None) -> str:
    if request is not None:
        app_mode = getattr(getattr(request, "app", None), "state", None)
        if app_mode is not None:
            value = getattr(request.app.state, "APP_MODE", None)
            if isinstance(value, str) and value:
                return value
    return OPENWEBUI_APP_MODE


def is_default_mode(request: Optional[Request] = None) -> bool:
    return get_app_mode(request) == "default"


def is_pageindex_mode(request: Optional[Request] = None) -> bool:
    return get_app_mode(request) == "pageindex"
