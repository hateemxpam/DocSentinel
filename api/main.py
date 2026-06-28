"""
main.py - Entry Point for DocSentinel FastAPI Application
===========================================================
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Initialize routes & initialization tasks
from api.routes import router
from cache.db import init_cache_table
from api.stats import init_stats_table

app = FastAPI(
    title="DocSentinel API",
    description="Intelligent Policy Compliance Assistant",
    version="1.0.0"
)

# 1. CORS Middleware Configurations (Allow all origins for demonstration purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Request Logging Middleware (Prints: method, endpoint, latency)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000
    print(f"[API Request] {request.method} {request.url.path} | Status: {response.status_code} | Latency: {latency_ms:.2f}ms")
    return response


# 3. Startup Events
@app.on_event("startup")
async def startup_event():
    print("\n--- DocSentinel API Starting Up ---")
    # Initialize cache and query stats tables in PostgreSQL
    init_cache_table()
    init_stats_table()
    print("DocSentinel API running.")


# 4. Include routing endpoints
app.include_router(router)
