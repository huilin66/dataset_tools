from __future__ import annotations
from pathlib import Path
import json
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse




def create_app(project_root: Path) -> FastAPI:
    app = FastAPI(title="BDD Facade Viewer")

    outputs = project_root / "outputs"
    web = project_root / "web"

    app.mount("/web", StaticFiles(directory=str(web), html=True), name="web")
    app.mount("/outputs", StaticFiles(directory=str(outputs)), name="outputs")

    @app.get("/")
    def root():
        return RedirectResponse(url="/web/")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/index")
    def get_index():
        p = outputs / "index.json"
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/poses")
    def get_poses():
        p = outputs / "poses_rgb.json"
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/pairs")
    def get_pairs():
        p = outputs / "pairs.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/image")
    def get_image(path: str):
        # 直接返回本地图片（注意：生产环境要加安全限制，这里先跑通）
        return FileResponse(path)

    return app
