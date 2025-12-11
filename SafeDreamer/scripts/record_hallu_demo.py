#!/usr/bin/env python
"""
Record a short clean vs hallucinated rollout in Safety-Gymnasium with the
GhostAwareWrapper. Intended as a minimal, standalone visual sanity check.

Examples (run from repo root):
  python SafeDreamer/scripts/record_hallu_demo.py --mode clean --steps 150 --output demo_clean_v2.mp4
  python SafeDreamer/scripts/record_hallu_demo.py --mode hallu --level 2 --steps 150 --output demo_hallu_level2_v2.mp4
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import imageio
import yaml

from SafeDreamer.embodied.envs.safetygymcoor import SafetyGymCoor
from envs.hallu_wrappers import GhostAwareWrapper


def load_hallu_cfg(level: int) -> Dict[str, Any]:
    """Load hallucination config from configs.yaml and normalize keys."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs.yaml"
    with cfg_path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    level_cfg = (
        raw.get("defaults", {})
        .get("hallu", {})
        .get("level_cfg", {})
    )
    # Convert string keys ("1","2","3") to ints for the wrapper.
    level_cfg_int = {int(k): v for k, v in level_cfg.items()}
    return {"level_cfg": level_cfg_int, "seed": raw.get("defaults", {}).get("seed", 0)}


def make_env(mode: str, level: int, cfg: Dict[str, Any]):
    env = SafetyGymCoor(
        "SafetyPointGoal1-v0",
        platform="gpu",
        mode="eval",  # eval mode always enables rendering inside SafetyGymCoor
        render=True,
    )
    if mode == "hallu":
        env = GhostAwareWrapper(env, level=level, cfg=cfg)
    return env


def main():
    parser = argparse.ArgumentParser(description="Record clean vs hallucinated demos.")
    parser.add_argument("--mode", choices=["clean", "hallu"], default="clean")
    parser.add_argument("--level", type=int, default=2, help="Hallucination level when mode=hallu")
    parser.add_argument("--steps", type=int, default=200, help="Number of env steps to record")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path")
    args = parser.parse_args()

    cfg = load_hallu_cfg(args.level)
    env = make_env(args.mode, args.level, cfg)

    out_path = args.output
    if out_path is None:
        suffix = f"_level{args.level}" if args.mode == "hallu" else ""
        out_path = f"demo_{args.mode}{suffix}_v2.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    try:
        obs = env.reset() if hasattr(env, "reset") else env.step({"reset": True})
        if isinstance(obs, dict):
            first = obs.get("image") or obs.get("image_far")
            if first is not None:
                frames.append(np.array(first, copy=True))

        for _ in range(args.steps):
            action = {k: space.sample() for k, space in env.act_space.items()}
            action["reset"] = False
            obs = env.step(action)
            frame = None
            if isinstance(obs, dict):
                frame = obs.get("image") or obs.get("image_far")
            if frame is not None:
                frames.append(np.array(frame, copy=True))

            if isinstance(obs, dict) and (obs.get("is_last") or obs.get("is_terminal") or obs.get("is_truncated")):
                obs = env.step({"reset": True})
    finally:
        if hasattr(env, "close"):
            env.close()

    if not frames:
        raise RuntimeError("No frames were captured; check render_mode and env setup.")

    imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"[record_hallu_demo] saved {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
