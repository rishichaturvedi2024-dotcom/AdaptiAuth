"""
AdaptiAuth — FastAPI Application Entry Point

Initializes the app, mounts routers, configures CORS and middleware,
and creates the database on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import auth, session, trust, dashboard, biometrics
from app.models.schemas import HealthResponse
from app.services.monitoring_loop import ContinuousMonitor

monitor = ContinuousMonitor()


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle hooks."""
    # Startup: initialize database schema
    init_db()
    print(f"✓ {settings.app_name} started in {settings.app_env} mode")
    print(f"✓ Database initialized at {settings.database_url}")
    
    # Start background loop
    await monitor.start()
    
    yield
    
    # Shutdown
    await monitor.stop()
    print(f"✗ {settings.app_name} shutting down")


# ── App factory ──────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    description=(
        "Risk-Adaptive Continuous Authentication Framework "
        "using Multi-Modal Biometrics and Explainable Trust Scoring"
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────
app.include_router(auth.router, prefix="/api")
app.include_router(session.router, prefix="/api")
app.include_router(trust.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(biometrics.router, prefix="/api")


# ── Health check ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check endpoint."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "app": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
