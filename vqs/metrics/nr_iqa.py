"""Optional GPU no-reference IQA (TOPIQ-NR via pyiqa)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NrIqaResult:
    mean: float | None
    median: float | None
    model: str
    device: str
    available: bool
    message: str = ""


class NrIqaScorer:
    """Lazy-loaded TOPIQ-NR (or fallback) scorer.

    TOPIQ-NR scores are higher-is-better roughly in 0..1.
    When pyiqa/torch are unavailable, ``score_frames`` returns None scores
    and classical-only decisions continue to work.
    """

    def __init__(self, model_name: str = "topiq_nr", device: str = "cpu", batch_size: int = 8):
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._metric = None
        self._torch = None
        self._error = ""
        self._tried = False

    def _load(self) -> bool:
        if self._tried:
            return self._metric is not None
        self._tried = True
        try:
            import torch
            import pyiqa

            self._torch = torch
            self._metric = pyiqa.create_metric(self.model_name, device=self.device)
            return True
        except Exception as exc:  # noqa: BLE001 — optional dependency path
            self._error = str(exc)
            self._metric = None
            return False

    @property
    def available(self) -> bool:
        return self._load()

    def score_bgr_frames(self, frames_bgr: list[np.ndarray], max_side: int = 512) -> NrIqaResult:
        if not frames_bgr:
            return NrIqaResult(None, None, self.model_name, self.device, False, "no frames")
        if not self._load():
            return NrIqaResult(
                None,
                None,
                self.model_name,
                self.device,
                False,
                self._error or "pyiqa/torch not available",
            )

        torch = self._torch
        scores: list[float] = []
        batch: list = []

        def flush():
            nonlocal batch, scores
            if not batch:
                return
            tensor = torch.stack(batch, dim=0).to(self.device)
            with torch.no_grad():
                out = self._metric(tensor)
            values = out.detach().float().cpu().view(-1).tolist()
            scores.extend(float(v) for v in values)
            batch = []

        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(1.0, float(max_side) / float(max(h, w)))
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            batch.append(tensor)
            if len(batch) >= self.batch_size:
                flush()
        flush()

        if not scores:
            return NrIqaResult(None, None, self.model_name, self.device, True, "empty scores")
        arr = np.asarray(scores, dtype=np.float64)
        return NrIqaResult(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            model=self.model_name,
            device=self.device,
            available=True,
        )


_SCORER_CACHE: dict[tuple, NrIqaScorer] = {}


def get_scorer(model_name: str, device: str, batch_size: int) -> NrIqaScorer:
    key = (model_name, device, batch_size)
    if key not in _SCORER_CACHE:
        _SCORER_CACHE[key] = NrIqaScorer(model_name, device, batch_size)
    return _SCORER_CACHE[key]
