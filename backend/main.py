import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models import registry
from backend.routes import meta, predict, analytics, site_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model artifacts...")
    status = registry.load_all()
    trained = [k for k, v in status.items() if v]
    if trained:
        log.info("Models loaded: %s", trained)
    else:
        log.warning(
            "No trained models found in %s. "
            "Run:  python -m scripts.train_models",
            settings.models_dir,
        )
    yield


app = FastAPI(
    title="Trial Duration Predictor",
    description="Predicts clinical trial duration from ClinicalTrials.gov data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(site_rates.router, prefix="/api")

# Serve frontend
_frontend = settings.frontend_dir
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(_frontend / "index.html"))
