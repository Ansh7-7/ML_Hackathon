"""HMM / Viterbi route reconstruction on top of the candidate classifier.

Hidden states
    At GPS observation i the hidden state is "the vehicle is on candidate arc c"
    for the top-k candidates of i (ranked by classifier probability). The arc id
    of every state is kept, so the decoded state maps 1:1 back to a road segment.

Emission
    log e_i(c) = w * log(max(p_i(c), floor)), where p_i(c) is the classifier's
    probability that c is the reference arc (w = emission_weight).

Transition (Newson & Krumm 2009 style)
    d_route = RoadNetwork.route_distance(a, t_a, b, t_b, cutoff): driving
              distance from the projection of obs i on arc a to the projection
              of obs i+1 on arc b along the DIRECTED network
    d_gc    = straight-line distance between the two GPS fixes
    log T(a->b) = -|d_route - d_gc| / beta_eff,
              beta_eff = beta * max(1, dt)^beta_dt_exp   (longer gaps -> looser)
    unreachable within cutoff = max(2*d_gc + slack, vmax*dt)  -> T = 0 (-inf).
    Backward movement along the same arc of up to back_tol_m is treated as GPS
    jitter of a slow or stationary vehicle (distance 0) instead of requiring a
    loop around the block; without this, slow stretches near a node pull the
    decoder onto the outgoing arc, where the projection is clamped at t = 0.
    A move onto the reverse twin of the same road (U-turn) gets an extra
    -uturn_penalty.

Special cases
    * observation with no candidate: it is skipped by the decoder (the chain
      continues from the previous observation that had candidates, with dt and
      d_gc measured between those two fixes); afterwards it inherits the
      previous decoded arc.
    * all transitions into observation i impossible (disconnected segments):
      the chain is broken, delta_i restarts from the emissions alone, and the
      break is logged (reason "disconnected").
    * time gap > gap_split_s: the chain is split on purpose (reason "gap").

Recurrence
    delta_0(c)  = log e_0(c)
    delta_i(c)  = log e_i(c) + max_c' [delta_{i-1}(c') + log T(c' -> c)]
    psi_i(c)    = argmax_c' [...]
    The best final state is traced back through psi; at a break the trace-back
    restarts from argmax delta of the segment that ends there.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .evaluate import fill_route


@dataclass
class HMMConfig:
    beta: float = 10.0
    emission_weight: float = 1.0
    top_k: int = 8
    prob_floor: float = 1e-4
    gap_split_s: float = 60.0
    beta_dt_exp: float = 0.5
    vmax_mps: float = 50.0
    cutoff_slack_m: float = 100.0
    uturn_penalty: float = 2.0
    back_tol_m: float = 5.0          # same-arc backward GPS jitter treated as standing still
    max_fill_m: float = 1000.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_lattice(obs, arc, t_proj, prob, n_obs, top_k):
    """Per observation: candidate arcs / t / prob sorted by prob (desc), capped at top_k.
    Returns list (len n_obs) of tuples (arcs, t, p) - empty arrays where no candidates."""
    obs = np.asarray(obs); arc = np.asarray(arc); t_proj = np.asarray(t_proj); prob = np.asarray(prob, float)
    order = np.lexsort((-prob, obs))
    o = obs[order]
    starts = np.searchsorted(o, np.arange(n_obs))
    ends = np.searchsorted(o, np.arange(n_obs), side="right")
    lat = []
    for i in range(n_obs):
        sl = order[starts[i]:min(ends[i], starts[i] + top_k)]
        lat.append((arc[sl], t_proj[sl], prob[sl]))
    return lat


def precompute_transitions(pt, lattice, cfg: HMMConfig):
    """Route distances between candidates of consecutive *active* observations.

    Depends only on the lattice and cutoff settings (not on beta / w), so it is
    computed once and reused while tuning the decoder."""
    net = pt.net
    active = np.array([i for i, c in enumerate(lattice) if len(c[0]) > 0], dtype=np.int64)
    steps = []
    for j in range(1, len(active)):
        i0, i1 = active[j - 1], active[j]
        A, TA, _ = lattice[i0]
        B, TB, _ = lattice[i1]
        gc = float(np.hypot(pt.px[i1] - pt.px[i0], pt.py[i1] - pt.py[i0]))
        dt = float(pt.t[i1] - pt.t[i0])
        if dt > cfg.gap_split_s:
            steps.append((gc, dt, None, None))
            continue
        cutoff = max(2.0 * gc + cfg.cutoff_slack_m, cfg.vmax_mps * dt)
        D = np.empty((len(A), len(B)))
        for ia in range(len(A)):
            for ib in range(len(B)):
                D[ia, ib] = net.route_distance(int(A[ia]), float(TA[ia]), int(B[ib]), float(TB[ib]), cutoff,
                                               back_tol=cfg.back_tol_m)
        uturn = (net.u[A][:, None] == net.v[B][None, :]) & (net.v[A][:, None] == net.u[B][None, :])
        steps.append((gc, dt, D, uturn))
    return active, steps


def _log_transition(D, uturn, gc, dt, cfg, ka, kb):
    D = D[:ka, :kb]
    beta_eff = cfg.beta * max(1.0, dt) ** cfg.beta_dt_exp
    logT = -np.abs(D - gc) / beta_eff
    if cfg.uturn_penalty:
        logT = logT - cfg.uturn_penalty * uturn[:ka, :kb]
    logT[~np.isfinite(D)] = -np.inf
    return logT


def viterbi(lattice, active, steps, cfg: HMMConfig, k=None):
    """Decode. `k` (<= the k used for the lattice) restricts to the top-k states.
    Returns (state index per active observation, list of (obs_index, reason) breaks)."""
    k = k or cfg.top_k
    if len(active) == 0:
        return np.empty(0, dtype=np.int64), []
    emis = [cfg.emission_weight * np.log(np.maximum(lattice[i][2][:k], cfg.prob_floor)) for i in active]
    delta = emis[0].copy()
    psi = [None]
    segments = []       # (end index, delta at the end of the segment)
    breaks = []
    for j in range(1, len(active)):
        gc, dt, D, uturn = steps[j - 1]
        if D is not None:
            cand = delta[:, None] + _log_transition(D, uturn, gc, dt, cfg, len(delta), len(emis[j]))
            best_prev = np.argmax(cand, axis=0)
            best = cand[best_prev, np.arange(cand.shape[1])]
        if D is None or not np.isfinite(best).any():
            # long time gap (deliberate split) or every transition impossible (disconnected
            # segments): close the current segment and restart from the emissions alone
            segments.append((j - 1, delta))
            breaks.append((int(active[j]), "gap" if D is None else "disconnected"))
            delta = emis[j].copy()
            psi.append(None)
            continue
        delta = emis[j] + best
        psi.append(best_prev)
    segments.append((len(active) - 1, delta))
    state = np.empty(len(active), dtype=np.int64)
    start = 0
    for e, dlt in segments:
        s = int(np.argmax(dlt))
        state[e] = s
        for j in range(e, start, -1):
            s = int(psi[j][s])
            state[j - 1] = s
        start = e + 1
    return state, breaks


def states_to_arcs(lattice, active, state, n_obs):
    """Decoded arc and projection t per observation; observations without candidates
    inherit the previous decoded arc (or the next one at the start of the trip)."""
    arcs = np.full(n_obs, -1, dtype=np.int64)
    ts = np.zeros(n_obs)
    for j, i in enumerate(active):
        arcs[i] = lattice[i][0][state[j]]
        ts[i] = lattice[i][1][state[j]]
    if len(active) and len(active) < n_obs:
        last = -1
        for i in range(n_obs):
            if arcs[i] >= 0:
                last = i
            elif last >= 0:
                arcs[i], ts[i] = arcs[last], ts[last]
        first = int(active[0])
        arcs[:first], ts[:first] = arcs[first], ts[first]
    return arcs, ts


def decode_trip(pt, obs, arc, t_proj, prob, cfg: HMMConfig, cache=None):
    """Full decode for one trip. Returns dict with per-observation arcs, the
    gap-filled route (arc sequence) and diagnostics. `cache` may hold a
    precomputed (lattice, active, steps) built with top_k >= cfg.top_k."""
    n = len(pt.px)
    if cache is None:
        lattice = build_lattice(obs, arc, t_proj, prob, n, cfg.top_k)
        active, steps = precompute_transitions(pt, lattice, cfg)
    else:
        lattice, active, steps = cache
    state, breaks = viterbi(lattice, active, steps, cfg, k=cfg.top_k)
    arcs, ts = states_to_arcs(lattice, active, state, n)
    route, unfilled = fill_route(pt.net, arcs, cfg.max_fill_m)
    return {"arcs": arcs, "t": ts, "route": route, "breaks": breaks, "unfillable_gaps": unfilled,
            "n_no_candidates": int(n - len(active))}
