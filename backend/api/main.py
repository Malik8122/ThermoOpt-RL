"""FastAPI backend for the ThermoOpt dashboard.

    python -m backend.api.main            # http://127.0.0.1:8000  (docs at /docs)

REST endpoints control one live digital twin and serve saved experiment results; ``/ws/simulation`` pushes the
full state after every simulated minute. The API holds no simulation logic: it forwards commands to
``SimulationService`` and reads result files through ``backend.api.results``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.api import results
from backend.api.service import SimulationService
from backend.config import PROJECT_ROOT, Config, get_config

MAX_BROADCASTS_PER_SECOND = 20.0


class ControllerRequest(BaseModel):
    controller: str


class ResetRequest(BaseModel):
    seed: int | None = None
    controller: str | None = None
    start_hour: float | None = Field(default=None, ge=0, lt=24)


class StepRequest(BaseModel):
    minutes: int = Field(default=1, ge=1, le=240)


class SpeedRequest(BaseModel):
    speed: float = Field(gt=0, le=120)


class WorkloadRequest(BaseModel):
    workload_type: Literal["NORMAL_CPU", "GPU_INFERENCE", "AI_TRAINING"] = "NORMAL_CPU"
    cpu: int = Field(default=20, ge=0, le=100)
    gpu: int = Field(default=0, ge=0, le=100)
    duration: int = Field(default=30, ge=1, le=240)
    slack: int = Field(default=30, ge=0, le=480)
    priority: int = Field(default=2, ge=1, le=3)


def _json(message: dict) -> str:
    def default(o):
        if hasattr(o, "item"):
            return o.item()
        return str(o)
    return json.dumps(message, default=default, allow_nan=False)


def _safe(message):
    """Replace non-finite floats so the payload is strict JSON."""
    if isinstance(message, float):
        return message if math.isfinite(message) else None
    if isinstance(message, dict):
        return {k: _safe(v) for k, v in message.items()}
    if isinstance(message, (list, tuple)):
        return [_safe(v) for v in message]
    if hasattr(message, "item"):
        return _safe(message.item())
    return message


def create_app(config: Config | None = None, autostart_loop: bool = True) -> FastAPI:
    config = config or get_config()
    service = SimulationService(config)
    clients: set[WebSocket] = set()

    async def broadcast(message: dict) -> None:
        text = _json(_safe(message))
        for ws in list(clients):
            try:
                await ws.send_text(text)
            except Exception:
                clients.discard(ws)

    async def run_loop() -> None:
        """While running: advance ``speed`` simulated minutes per second, broadcasting at most 20 times a second."""
        while True:
            if not service.running:
                await asyncio.sleep(0.05)
                continue
            rate = min(service.speed, MAX_BROADCASTS_PER_SECOND)
            steps = max(1, round(service.speed / rate))
            message = service.step_many(steps)
            await broadcast(message)
            await asyncio.sleep(1.0 / rate)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(run_loop()) if autostart_loop else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="ThermoOpt-RL API", version="1.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=config.api.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])

    def ok() -> dict:
        return _safe(service.state())

    # ------------------------------------------------------------- live twin
    @app.get("/api/status")
    def status() -> dict:
        return ok()

    @app.get("/api/racks")
    def racks() -> list[dict]:
        return _safe(service.state()["racks"])

    @app.get("/api/workloads")
    def workloads() -> dict:
        s = service.state()
        running = [j for r in s["racks"] for j in r["jobs"]]
        return _safe({"queue": s["queue"], "running": running, "finished": s["finished"],
                      "decisions": list(service.decisions), "counters": s["counters"]})

    @app.get("/api/history")
    def history() -> dict:
        return _safe({"history": list(service.history), "decisions": list(service.decisions)})

    @app.get("/api/controllers")
    def controllers() -> list[dict]:
        return service.controllers()

    @app.post("/api/simulation/start")
    async def start() -> dict:
        service.running = True
        await broadcast({"type": "control", "running": True, "speed": service.speed})
        return ok()

    @app.post("/api/simulation/pause")
    async def pause() -> dict:
        service.running = False
        await broadcast({"type": "control", "running": False, "speed": service.speed})
        return ok()

    @app.post("/api/simulation/speed")
    async def speed(req: SpeedRequest) -> dict:
        service.speed = req.speed
        await broadcast({"type": "control", "running": service.running, "speed": service.speed})
        return ok()

    @app.post("/api/simulation/reset")
    async def reset(req: ResetRequest) -> dict:
        try:
            service.reset(seed=req.seed, controller=req.controller, start_hour=req.start_hour)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        await broadcast(service.init_message())
        return ok()

    @app.post("/api/simulation/step")
    async def step(req: StepRequest) -> dict:
        message = service.step_many(req.minutes)
        await broadcast(service.init_message() if req.minutes > 1 else message)
        return ok()

    @app.post("/api/simulation/controller")
    async def controller(req: ControllerRequest) -> dict:
        try:
            service.set_controller(req.controller)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        await broadcast({"type": "controller", "controller": service.controller_key,
                         "controllers": service.controllers(), "state": _safe(service.state())})
        return ok()

    @app.post("/api/workload")
    async def workload(req: WorkloadRequest) -> dict:
        if req.cpu == 0 and req.gpu == 0:
            raise HTTPException(status_code=400, detail="a job needs CPU or GPU")
        result = service.inject_workload(**req.model_dump())
        await broadcast({"type": "state", "state": _safe(service.state())})
        return _safe(result)

    @app.get("/api/predictions")
    def predictions() -> dict:
        s = service.state()
        return _safe({"horizons": s["horizons"], "available": service.forecaster is not None,
                      "racks": [{"name": r["name"], "temperature": r["temperature"], "forecast": r["forecast"]}
                                for r in s["racks"]]})

    # --------------------------------------------------------- saved results
    @app.get("/api/meta")
    def meta() -> dict:
        return results.meta(config)

    @app.get("/api/comparison")
    def comparison() -> dict:
        return results.comparison(config)

    @app.get("/api/training/{algorithm}/metrics")
    def training(algorithm: str) -> dict:
        data = results.training(config, algorithm)
        if data is None:
            raise HTTPException(status_code=404, detail=f"no training runs for {algorithm!r}")
        return data

    @app.get("/api/predictions/summary")
    def prediction_summary() -> dict:
        data = results.prediction(config)
        if data is None:
            raise HTTPException(status_code=404, detail="temperature forecaster has not been evaluated")
        return data

    # ------------------------------------------------------------ websocket
    @app.websocket("/ws/simulation")
    async def ws_simulation(ws: WebSocket) -> None:
        await ws.accept()
        clients.add(ws)
        try:
            await ws.send_text(_json(_safe(service.init_message())))
            while True:
                await ws.receive_text()          # clients only listen; this detects disconnects
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    # ----------------------------------------------- built frontend (if any)
    dist = PROJECT_ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            target = dist / path
            return FileResponse(target if path and target.is_file() else dist / "index.html")

    return app


def main() -> None:
    import uvicorn

    cfg = get_config()
    uvicorn.run(create_app(cfg), host=cfg.api.host, port=cfg.api.port, log_level="info")


if __name__ == "__main__":
    main()
