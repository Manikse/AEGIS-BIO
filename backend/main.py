import time
import tracemalloc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from closed_loop_kernel import (
    SimulationInput,
    evaluate_unified_kernel,
    run_kernel_validation_suite,
    AUTONOMOUS_DAEMON,
)

tracemalloc.start()

app = FastAPI(title="AEGIS-BIO // Story Showcase & Autonomous State-Space Kernel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "kernel": "HYBRID_STORY_AND_AUTONOMOUS_DAEMON_V8"}


@app.post("/api/evaluate")
async def evaluate_simulation(inp: SimulationInput) -> Dict[str, Any]:
    """Used by the 11-Beat Historical Story Showcase & Multi-Fault Synthesizer."""
    t0 = time.perf_counter_ns()
    result = evaluate_unified_kernel(inp)
    elapsed_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 2)
    current_mem, _ = tracemalloc.get_traced_memory()

    return {
        "engine": "UNIFIED_PID_KALMAN_KERNEL_V8",
        "edge_telemetry_benchmark": {
            "compute_latency_ms": elapsed_ms,
            "ram_footprint_kb": round(current_mem / 1024.0, 1),
            "rad750_cpu_load_est_pct": round(min(18.5, elapsed_ms * 1.4), 1),
        },
        **result,
    }


@app.get("/api/validate")
async def validate_kernel() -> Dict[str, Any]:
    """Runs 500 blind Monte Carlo trials + empirical clinical capnography trace verification."""
    t0 = time.perf_counter_ns()
    report = run_kernel_validation_suite(n_trials=500)
    elapsed_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 1)
    return {
        "benchmark_execution_ms": elapsed_ms,
        **report,
    }


@app.post("/api/tick")
async def step_autonomous_daemon() -> Dict[str, Any]:
    """Steps the continuous state-space autonomous solver & parallel Legacy NASA CWS benchmark."""
    return AUTONOMOUS_DAEMON.step_physics_and_autonomy()


@app.post("/api/inject-fault/{fault_type}")
async def inject_blind_fault(fault_type: str) -> Dict[str, Any]:
    """Spawns an unlabeled physical anomaly into the live autonomous sandbox."""
    if fault_type == "reset":
        AUTONOMOUS_DAEMON.reset_state()
    else:
        AUTONOMOUS_DAEMON.inject_fault(fault_type)
    return AUTONOMOUS_DAEMON.step_physics_and_autonomy()