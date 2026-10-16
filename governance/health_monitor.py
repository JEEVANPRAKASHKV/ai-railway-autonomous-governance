"""
governance/health_monitor.py
────────────────────────────
Pillar 1 – System Health Monitor & SLA Compliance Tracker.

Tracks per-module:
  • Request counts, error counts, latency distributions
  • P50 / P95 / P99 latency percentiles
  • SLA breach detection (configurable thresholds)
  • Circuit-breaker state (CLOSED → OPEN → HALF-OPEN)
  • Uptime counters
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from governance.telemetry_bus import telemetry_bus


# ──────────────────────────────────────────────────────────────────────────────
# Constants & SLA defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SLA = {
    "p95_latency_ms": 3000,   # 3 seconds
    "error_rate_pct": 5.0,    # 5%
    "min_uptime_pct": 99.0,   # 99% uptime
}

CIRCUIT_FAILURE_THRESHOLD = 5    # consecutive errors to OPEN circuit
CIRCUIT_RECOVERY_TIMEOUT_S = 60  # seconds before attempting HALF-OPEN
LATENCY_WINDOW_SIZE = 500        # rolling window for latency samples


# ──────────────────────────────────────────────────────────────────────────────
# Circuit Breaker State
# ──────────────────────────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "CLOSED"          # Normal operation
    OPEN = "OPEN"              # Blocking requests – too many failures
    HALF_OPEN = "HALF_OPEN"    # Probing – allowing one request through


@dataclass
class CircuitBreaker:
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_opened_at: Optional[float] = None

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD:
            self.state = CircuitState.OPEN
            self.last_opened_at = time.monotonic()

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self.last_opened_at or 0)
            if elapsed >= CIRCUIT_RECOVERY_TIMEOUT_S:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow exactly one probe
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Per-module Stats
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleStats:
    module_id: str
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW_SIZE))
    circuit: CircuitBreaker = field(default_factory=CircuitBreaker)
    started_at: float = field(default_factory=time.monotonic)
    sla_breaches: int = 0
    last_request_at: Optional[float] = None

    # ── derived metrics ──────────────────────────────────────────────────────

    @property
    def error_rate_pct(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_errors / self.total_requests * 100, 2)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round(self.total_latency_ms / self.total_requests, 2)

    def _percentile(self, p: float) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * p / 100)
        return round(sorted_samples[min(idx, len(sorted_samples) - 1)], 2)

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def uptime_pct(self) -> float:
        elapsed = time.monotonic() - self.started_at
        if elapsed <= 0:
            return 100.0
        # Simplified: uptime degrades proportional to open-circuit time
        circuit_down = self.circuit.last_opened_at or 0
        return round(max(0.0, 100.0 - (circuit_down / elapsed * 100)), 2)

    def sla_status(self, sla: dict = DEFAULT_SLA) -> dict:
        breaches = []
        if self.p95_ms > sla["p95_latency_ms"]:
            breaches.append(f"p95 latency {self.p95_ms}ms > {sla['p95_latency_ms']}ms SLA")
        if self.error_rate_pct > sla["error_rate_pct"]:
            breaches.append(f"error rate {self.error_rate_pct}% > {sla['error_rate_pct']}% SLA")
        return {
            "compliant": len(breaches) == 0,
            "breaches": breaches,
        }

    def to_dict(self) -> dict:
        return {
            "module_id": self.module_id,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate_pct": self.error_rate_pct,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "uptime_pct": self.uptime_pct,
            "circuit_state": self.circuit.state.value,
            "sla_breaches_total": self.sla_breaches,
            "last_request_at": (
                datetime.fromtimestamp(self.last_request_at, tz=timezone.utc).isoformat()
                if self.last_request_at
                else None
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Health Monitor (Singleton)
# ──────────────────────────────────────────────────────────────────────────────

class HealthMonitor:
    """Thread-safe, singleton health monitor for all platform modules."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._modules: Dict[str, ModuleStats] = {}
        self._sla_config: dict = DEFAULT_SLA.copy()

    # ── Registration ─────────────────────────────────────────────────────────

    def _get_or_create(self, module_id: str) -> ModuleStats:
        if module_id not in self._modules:
            self._modules[module_id] = ModuleStats(module_id=module_id)
        return self._modules[module_id]

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_request(
        self,
        module_id: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        with self._lock:
            stats = self._get_or_create(module_id)
            stats.total_requests += 1
            stats.total_latency_ms += latency_ms
            stats.latency_samples.append(latency_ms)
            stats.last_request_at = time.time()

            if success:
                stats.circuit.record_success()
            else:
                stats.total_errors += 1
                stats.circuit.record_failure()

            # SLA breach check
            sla_result = stats.sla_status(self._sla_config)
            if not sla_result["compliant"]:
                stats.sla_breaches += 1
                telemetry_bus.publish(
                    module_id=module_id,
                    event_type="SLA_BREACH",
                    payload={
                        "breaches": sla_result["breaches"],
                        "p95_ms": stats.p95_ms,
                        "error_rate_pct": stats.error_rate_pct,
                    },
                    severity="WARNING",
                )

    def allow_request(self, module_id: str) -> bool:
        """Check circuit breaker before forwarding a request."""
        with self._lock:
            stats = self._get_or_create(module_id)
            return stats.circuit.allow_request()

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_module_health(self, module_id: str) -> dict:
        with self._lock:
            stats = self._get_or_create(module_id)
            result = stats.to_dict()
            result["sla"] = stats.sla_status(self._sla_config)
            return result

    def get_platform_health(self) -> dict:
        with self._lock:
            modules = {mid: s.to_dict() for mid, s in self._modules.items()}
            total_req = sum(s.total_requests for s in self._modules.values())
            total_err = sum(s.total_errors for s in self._modules.values())
            open_circuits = [
                mid for mid, s in self._modules.items()
                if s.circuit.state != CircuitState.CLOSED
            ]
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "platform_status": "DEGRADED" if open_circuits else "HEALTHY",
                "total_requests": total_req,
                "total_errors": total_err,
                "overall_error_rate_pct": (
                    round(total_err / total_req * 100, 2) if total_req else 0.0
                ),
                "open_circuits": open_circuits,
                "modules": modules,
            }

    def get_sla_report(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "module_id": mid,
                    **s.sla_status(self._sla_config),
                    "p95_ms": s.p95_ms,
                    "error_rate_pct": s.error_rate_pct,
                    "sla_breaches_total": s.sla_breaches,
                }
                for mid, s in self._modules.items()
            ]

    def update_sla_config(self, config: dict) -> None:
        with self._lock:
            self._sla_config.update(config)

    def reset_module(self, module_id: str) -> None:
        with self._lock:
            if module_id in self._modules:
                self._modules[module_id] = ModuleStats(module_id=module_id)

    def reset_all(self) -> None:
        with self._lock:
            for mid in list(self._modules.keys()):
                self._modules[mid] = ModuleStats(module_id=mid)


# ── Singleton export ──────────────────────────────────────────────────────────
health_monitor = HealthMonitor()
