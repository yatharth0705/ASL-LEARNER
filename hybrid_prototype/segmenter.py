import math
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple, Dict, Any
import numpy as np

from landmarks import (
    LEFT_HAND_START,
    RIGHT_HAND_START,
    N_HAND,
    POSE_START,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    hands_missing,
)

@dataclass
class SegmenterOptions:
    gap_ms: float = 300.0
    move_speed: float = 0.6
    min_frames: int = 5
    max_ms: float = 4000.0
    preroll_ms: float = 200.0
    leave_ms: float = 500.0

@dataclass
class Segment:
    frames: List[np.ndarray]
    timestamps: List[float]
    fps: float
    duration_ms: float

@dataclass
class Sample:
    frame: np.ndarray
    t: float

HAND_STARTS = [LEFT_HAND_START, RIGHT_HAND_START]

def compute_shoulder_anchor(frame: np.ndarray) -> Tuple[float, float, float]:
    """Midpoint (cx, cy) and width of shoulders in a single frame."""
    l_idx = (POSE_START + POSE_LEFT_SHOULDER) * 3
    r_idx = (POSE_START + POSE_RIGHT_SHOULDER) * 3
    lx, ly = frame[l_idx], frame[l_idx + 1]
    rx, ry = frame[r_idx], frame[r_idx + 1]
    
    if np.isfinite(lx) and np.isfinite(ly) and np.isfinite(rx) and np.isfinite(ry):
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0
        scale = max(math.hypot(lx - rx, ly - ry), 1e-3)
        return cx, cy, scale
    return 0.5, 0.5, 0.25

def hand_speed(prev: Sample, next_s: Sample) -> float:
    """Mean displacement of hand points seen in both frames, in shoulder widths per second."""
    dt = (next_s.t - prev.t) / 1000.0
    if not (dt > 0):
        return 0.0
    _, _, scale = compute_shoulder_anchor(next_s.frame)
    total = 0.0
    count = 0
    for start in HAND_STARTS:
        for i in range(N_HAND):
            k = (start + i) * 3
            p_dx = prev.frame[k]
            p_dy = prev.frame[k + 1]
            n_dx = next_s.frame[k]
            n_dy = next_s.frame[k + 1]
            if np.isfinite(p_dx) and np.isfinite(n_dx) and np.isfinite(p_dy) and np.isfinite(n_dy):
                total += math.hypot(n_dx - p_dx, n_dy - p_dy)
                count += 1
    return (total / count / scale / dt) if count else 0.0

class Segmenter:
    def __init__(self, opts: Optional[SegmenterOptions] = None):
        self.opts = opts or SegmenterOptions()
        self.samples: List[Sample] = []
        self.state: Literal['idle', 'signing', 'resting'] = 'idle'
        self.still_since: Optional[float] = None
        self.absent_since: Optional[float] = None
        self.sign_start: Optional[float] = None
        self.sign_still: bool = False
        self.last_kept_count: int = 0

    def reset(self):
        self.samples.clear()
        self.state = 'idle'
        self.still_since = None
        self.absent_since = None
        self.sign_start = None
        self.sign_still = False
        self.last_kept_count = 0

    def push(self, frame: np.ndarray, t: float, crowded: bool = False) -> Optional[Dict[str, Any]]:
        if crowded:
            res = self.finish('absent', t)
            self.reset()
            return res

        prev = self.samples[-1] if self.samples else None
        sample = Sample(frame, t)
        self.samples.append(sample)

        # Drop samples older than max(gap, preroll) while idle/resting
        if self.state != 'signing':
            horizon = t - max(self.opts.gap_ms, self.opts.preroll_ms) - 100.0
            idx = 0
            while idx < len(self.samples) - 1 and self.samples[idx].t < horizon:
                idx += 1
            if idx > 0:
                self.samples = self.samples[idx:]

        absent = hands_missing(frame)
        if absent:
            if self.absent_since is None:
                self.absent_since = t
            self.still_since = None
        else:
            self.absent_since = None
            speed = hand_speed(prev, sample) if prev else 0.0
            if speed < self.opts.move_speed:
                if self.still_since is None:
                    self.still_since = t
            else:
                self.still_since = None

        if self.state == 'idle':
            if not absent and self.still_since is None and prev:
                self.state = 'signing'
                self.sign_start = self.preroll_start(t)
                self.sign_still = False
                return {'type': 'start', 't': self.sign_start}
            return None

        if self.state == 'resting':
            if self.absent_since is not None and (t - self.absent_since >= self.opts.leave_ms):
                self.state = 'idle'
            elif not absent and self.still_since is None and prev:
                self.state = 'signing'
                self.sign_start = self.preroll_start(t)
                self.sign_still = False
                return {'type': 'start', 't': self.sign_start}
            return None

        # State is 'signing'
        # Apply small epsilon (0.1ms) to guard against 30 FPS floating point roundoff (e.g. 299.9999999999993 vs 300.0)
        if self.sign_start is not None and (t - self.sign_start >= self.opts.max_ms - 1e-1):
            return self.finish('max', t)
        if self.absent_since is not None and (t - self.absent_since >= self.opts.gap_ms - 1e-1):
            return self.finish('absent', t)
        if self.still_since is not None and (t - self.still_since >= self.opts.gap_ms - 1e-1):
            return self.finish('still', t)

        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        if self.state != 'signing':
            return None
        t = self.samples[-1].t if self.samples else 0.0
        return self.finish('flush', t)

    def preroll_start(self, now: float) -> float:
        cutoff = now - self.opts.preroll_ms
        for s in reversed(self.samples):
            if s.t < cutoff:
                return s.t
        return self.samples[0].t if self.samples else now

    def finish(self, reason: Literal['absent', 'still', 'max', 'flush'], now: float) -> Optional[Dict[str, Any]]:
        start_t = self.sign_start or (self.samples[0].t if self.samples else now)
        was_still = self.sign_still

        cut_t = now
        if reason == 'absent' and self.absent_since is not None:
            cut_t = self.absent_since
        elif reason == 'still' and self.still_since is not None:
            cut_t = self.still_since

        kept = [s for s in self.samples if s.t >= start_t and s.t <= cut_t and not hands_missing(s.frame)]
        self.last_kept_count = len(kept)

        # Update state
        self.state = 'resting' if reason == 'still' else 'idle'
        self.sign_start = None
        self.sign_still = (reason == 'still')
        self.still_since = None
        self.absent_since = None
        self.samples.clear()

        # Check for hands dropping discard after still hold
        if reason == 'absent' and was_still and (cut_t - start_t < self.opts.leave_ms):
            return {'type': 'discard', 'reason': 'hands-leaving'}

        if len(kept) < self.opts.min_frames:
            return {'type': 'discard', 'reason': 'too-short'}

        timestamps = [s.t for s in kept]
        frames = [s.frame for s in kept]
        duration_ms = max(timestamps[-1] - timestamps[0], 1.0)
        fps = (len(kept) - 1) / (duration_ms / 1000.0) if len(kept) > 1 else 30.0
        fps = max(5.0, min(fps, 120.0))

        segment = Segment(frames=frames, timestamps=timestamps, fps=fps, duration_ms=duration_ms)
        return {'type': 'end', 'reason': reason, 'segment': segment}

