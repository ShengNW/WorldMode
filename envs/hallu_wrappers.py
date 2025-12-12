import numpy as np
from typing import Any, Dict, Optional

import embodied


class GhostAwareWrapper(embodied.core.base.Wrapper):
    """
    Injects ghost obstacles (FP) and hides some hazards (FN) in observation vectors.

    Designed for SafetyGymCoor-style envs where the flattened observation contains
    slices for `hazards` and optionally `robot` coordinates recorded in
    `env.key_to_slice`. The wrapper keeps the observation shape unchanged by
    reusing existing hazard slots and filling them with synthetic coordinates or
    zeroing them out to simulate missing detections.
    """

    def __init__(self, env, level: int = 1, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        self.level = int(level)
        self.cfg = cfg or {}
        self.rng = np.random.default_rng(self.cfg.get("seed"))
        self._hazard_slice = None
        self._hazard_slots = 0
        self._robot_slice = None
        self._init_layout_from_env()
        self._load_level_params()
        self._reset_hallu_state()

    def _init_layout_from_env(self) -> None:
        if hasattr(self.env, "key_to_slice"):
            key_to_slice = getattr(self.env, "key_to_slice", {}) or {}
            if "hazards" in key_to_slice:
                self._hazard_slice = key_to_slice["hazards"]
                self._hazard_slots = max(1, (self._hazard_slice.stop - self._hazard_slice.start) // 2)
            if "robot" in key_to_slice:
                self._robot_slice = key_to_slice["robot"]

    def _load_level_params(self) -> None:
        default_level_cfg = {
            1: {"fp_prob": 0.1, "fn_prob": 0.0, "min_frames": 1, "max_frames": 2, "max_ghosts": 1},
            2: {"fp_prob": 0.3, "fn_prob": 0.05, "min_frames": 3, "max_frames": 5, "max_ghosts": 2},
            3: {"fp_prob": 0.5, "fn_prob": 0.1, "min_frames": 5, "max_frames": 10, "max_ghosts": 3},
        }
        level_cfg = (self.cfg.get("level_cfg") or default_level_cfg).get(
            self.level, default_level_cfg[1]
        )
        self.fp_prob = float(level_cfg.get("fp_prob", 0.0))
        self.fn_prob = float(level_cfg.get("fn_prob", 0.0))
        self.min_frames = int(level_cfg.get("min_frames", 1))
        self.max_frames = int(level_cfg.get("max_frames", self.min_frames))
        self.max_frames = max(self.min_frames, self.max_frames)
        self.max_ghosts = int(level_cfg.get("max_ghosts", 2))

    def _reset_hallu_state(self) -> None:
        self._fp_state = None
        self._fn_state = None

    def reset(self, *args, **kwargs):  # type: ignore
        if hasattr(self.env, "reset"):
            obs = self.env.reset(*args, **kwargs)
        else:
            obs = self.env.step({"reset": True})
        self._reset_hallu_state()
        return self._apply_hallu(obs)

    def step(self, action):  # type: ignore
        obs = self.env.step(action)
        return self._apply_hallu(obs)

    def _maybe_init_layout(self, obs: Any) -> None:
        if self._hazard_slice is not None:
            return
        if not isinstance(obs, dict) or "observation" not in obs:
            return
        if hasattr(self.env, "key_to_slice"):
            key_to_slice = getattr(self.env, "key_to_slice", {}) or {}
            if "hazards" in key_to_slice:
                self._hazard_slice = key_to_slice["hazards"]
                self._hazard_slots = max(1, (self._hazard_slice.stop - self._hazard_slice.start) // 2)
            if "robot" in key_to_slice:
                self._robot_slice = key_to_slice["robot"]

    def _apply_hallu(self, obs: Any):
        if not isinstance(obs, dict) or "observation" not in obs:
            return obs
        self._maybe_init_layout(obs)
        if self._hazard_slice is None or self._hazard_slots <= 0:
            return obs

        obs_vec = np.array(obs["observation"], copy=True, dtype=np.float32)
        hazard_view = obs_vec[self._hazard_slice]
        hazards = hazard_view.reshape(self._hazard_slots, 2)

        self._maybe_start_fp(obs_vec)
        self._maybe_start_fn()

        hallu_hazards = hazards.copy()
        base_mask = np.ones(self._hazard_slots, dtype=bool)
        ghost_indices = []

        if self._fn_state is not None:
            mask = self._fn_state["mask"]
            hallu_hazards[mask] = 0.0
            base_mask[mask] = False
            self._fn_state["frames_left"] -= 1
            if self._fn_state["frames_left"] <= 0:
                self._fn_state = None

        if self._fp_state is not None:
            ghosts = self._fp_state["coords"]
            hallu_hazards, ghost_indices = self._inject_ghosts(hallu_hazards, ghosts)
            self._fp_state["frames_left"] -= 1
            if self._fp_state["frames_left"] <= 0:
                self._fp_state = None

        obs_vec[self._hazard_slice] = hallu_hazards.astype(np.float32).flatten()
        ghost_mask = base_mask.astype(np.float32)
        ghost_labels = base_mask.astype(np.float32)
        for idx in ghost_indices:
            ghost_labels[idx] = 0.0
            ghost_mask[idx] = 1.0

        obs_out = dict(obs)
        obs_out["observation"] = obs_vec
        obs_out["ghost_labels"] = ghost_labels
        obs_out["ghost_mask"] = ghost_mask
        return obs_out

    def _maybe_start_fp(self, obs_vec: np.ndarray) -> None:
        if self.fp_prob <= 0 or self._fp_state is not None:
            return
        if self.rng.random() >= self.fp_prob:
            return
        frames = int(self.rng.integers(self.min_frames, self.max_frames + 1))
        count = max(1, min(self.max_ghosts, self._hazard_slots))
        center = np.zeros(2, dtype=np.float32)
        if self._robot_slice is not None:
            center = np.array(obs_vec[self._robot_slice][:2], dtype=np.float32)
        ghosts = self._sample_ghosts(count, center)
        self._fp_state = {"coords": ghosts, "frames_left": frames}

    def _maybe_start_fn(self) -> None:
        if self.fn_prob <= 0 or self._fn_state is not None:
            return
        if self.rng.random() >= self.fn_prob:
            return
        frames = int(self.rng.integers(self.min_frames, self.max_frames + 1))
        mask = np.zeros(self._hazard_slots, dtype=bool)
        # Hide at least one hazard to make the effect visible.
        idx = self.rng.integers(0, self._hazard_slots)
        mask[idx] = True
        self._fn_state = {"mask": mask, "frames_left": frames}

    def _sample_ghosts(self, count: int, center: np.ndarray) -> np.ndarray:
        # Sample ghosts around the robot within a modest radius.
        radius = self.cfg.get("fp_radius", 1.5)
        ghosts = self.rng.uniform(-radius, radius, size=(count, 2)).astype(np.float32)
        return ghosts + center.reshape(1, 2)

    def _inject_ghosts(self, hazards: np.ndarray, ghosts: np.ndarray):
        hallu = hazards.copy()
        available = list(np.where(np.all(hallu == 0.0, axis=1))[0])
        used_indices = []
        for g in ghosts:
            if available:
                idx = available.pop(0)
            else:
                idx = int(self.rng.integers(0, hallu.shape[0]))
            hallu[idx] = g
            used_indices.append(idx)
        return hallu, used_indices

    @property
    def obs_space(self):
        spaces = dict(self.env.obs_space)
        if self._hazard_slice is None:
            self._init_layout_from_env()
        if self._hazard_slice is not None and self._hazard_slots > 0:
            spaces["ghost_labels"] = embodied.Space(np.float32, (self._hazard_slots,), 0.0, 1.0)
            spaces["ghost_mask"] = embodied.Space(np.float32, (self._hazard_slots,), 0.0, 1.0)
        return spaces


__all__ = ["GhostAwareWrapper"]
