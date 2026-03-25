from __future__ import annotations
from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


def guess_project_root() -> Path:
    """
    server.py 位于: <root>/src/bddfacade/server.py
    所以往上两级就是 <root>/src，再往上一级是 <root>
    """
    return Path(__file__).resolve().parents[2]  # <root>


def create_app(project_root: Path | None = None) -> FastAPI:
    project_root = Path(project_root) if project_root else guess_project_root()
    app = FastAPI(title="BDD Facade Viewer")

    outputs = project_root / "outputs"
    web = project_root / "web"

    # 静态资源
    app.mount("/web", StaticFiles(directory=str(web), html=True), name="web")
    app.mount("/outputs", StaticFiles(directory=str(outputs)), name="outputs")

    @app.get("/")
    def root():
        return RedirectResponse(url="/web/")

    @app.get("/api/health")
    def health():
        return {"ok": True, "project_root": str(project_root)}

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
        # 跑通优先：直接返回本地路径
        return FileResponse(path)

    return app


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--root", default=None, help="project root, default auto")
    args = parser.parse_args()

    app = create_app(args.root)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
