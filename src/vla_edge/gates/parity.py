"""Parity metrics.

Deliberately small and dependency-free at the top level, so the thresholds can
be reasoned about and unit-tested without a GPU or a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParityResult:
    name: str
    rel_rms: float
    max_abs: float
    threshold: float
    passed: bool

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}  {self.name:<28} rel_rms={self.rel_rms:.5f} "
            f"max_abs={self.max_abs:.5f} (threshold {self.threshold:.5f})"
        )


def rel_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Relative RMS error, normalized by the reference's own RMS.

    Normalizing matters: absolute error on activations whose scale varies by
    orders of magnitude between stages is not comparable across stages, and a
    single absolute threshold would be simultaneously too strict for one stage
    and useless for another.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    # Compare shapes BEFORE flattening. A transposed output is one of the most
    # common compiled-backend bugs, and (2,3) vs (3,2) ravel to the same
    # shape. Checking after the ravel would let exactly that class through.
    if cand.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: candidate {cand.shape} vs reference {ref.shape}"
        )
    cand, ref = cand.ravel(), ref.ravel()
    denom = float(np.sqrt(np.mean(ref**2)))
    if denom == 0.0:
        return float(np.sqrt(np.mean(cand**2)))
    return float(np.sqrt(np.mean((cand - ref) ** 2)) / denom)


def check_stage(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    threshold: float,
) -> ParityResult:
    """Compare one stage against its recorded reference.

    Pick ``threshold`` from your measured noise floor, not from taste. Run the
    reference against itself in the trained precision and use that number; on
    the checkpoints we have measured it is 0.04-0.06 for bf16. A stage an
    order of magnitude above the floor is broken even when the robot appears
    to behave. The canonical example is an fp16 language backbone, which
    lands at 0.55 and raises nothing.
    """
    value = rel_rms(candidate, reference)
    max_abs = float(
        np.max(np.abs(np.asarray(candidate, dtype=np.float64)
                      - np.asarray(reference, dtype=np.float64)))
    )
    return ParityResult(name, value, max_abs, threshold, value <= threshold)


def seed_spread_gate(
    candidate_same_seed: np.ndarray,
    reference_same_seed: np.ndarray,
    reference_other_seed: np.ndarray,
    ratio: float = 0.5,
) -> ParityResult:
    """End-to-end gate scaled to the policy's own stochasticity.

    A flow-matching policy is stochastic: two seeds legitimately disagree, so
    an absolute action tolerance is meaningless. The natural scale is the
    policy disagreeing with *itself*.

    1. spread   = max|reference(seed A) - reference(seed B)|
    2. deviation = max|candidate(seed A) - reference(seed A)|
    3. pass if deviation <= ratio * spread

    If this gate is flaky, suspect the harness before the backend: an
    integrator that mutates the caller's noise tensor in place produces
    "parity failures" that are really input corruption, and they move around
    between runs.
    """
    spread = float(
        np.max(np.abs(np.asarray(reference_same_seed, dtype=np.float64)
                      - np.asarray(reference_other_seed, dtype=np.float64)))
    )
    deviation = float(
        np.max(np.abs(np.asarray(candidate_same_seed, dtype=np.float64)
                      - np.asarray(reference_same_seed, dtype=np.float64)))
    )
    threshold = ratio * spread
    # A degenerate spread means the two seeds produced identical actions,
    # which for a stochastic policy means the seeding is not wired up. Fail
    # rather than divide by ~zero and report a meaningless pass.
    if spread <= 1e-9:
        return ParityResult(
            "e2e-seed-spread", deviation, deviation, 0.0, False
        )
    return ParityResult(
        "e2e-seed-spread", deviation / spread, deviation, ratio,
        deviation <= threshold,
    )
