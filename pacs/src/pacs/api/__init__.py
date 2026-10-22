"""api 包：本地 Web 服务（零第三方依赖，标准库 http.server）。"""

from .server import PacsHandler, serve

__all__ = ["PacsHandler", "serve"]
