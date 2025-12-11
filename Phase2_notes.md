# Phase 2 baseline runbook

## Environment and hallucination settings
- Task: `SafetyPointGoal1-v0` via `SafetyGymCoor` (vector obs, repeat 5, camera `fixedfar`).
- Hallucination Level2 (used for hallu baselines): `fp_prob=0.3`, `fn_prob=0.05`, `min_frames=3`, `max_frames=5`, `max_ghosts=2`, `fp_radius=1.5` (default). FP ghosts persist 3–5 frames around the robot, often forming a pseudo ring; FN hides at least one real hazard. Expectation: with strict cost limit, planner frequently sees dense obstacles ⇒ near-frozen; with relaxed cost limit, ghost-induced caution drops and real hazard FN causes collisions.

## Baseline configs (combine with `osrp_vector` + `osrp_vector_phase2_common`)
- `osrp_vector_unsafe`: ignores cost (`use_cost=False`, huge cost_limit), hallu off.
- `osrp_vector_safedreamer_clean`: safe dreamer on clean env (`use_cost=True`, `cost_limit=2.0`), hallu off.
- `osrp_vector_safedreamer_hallu_strict`: hallu Level2 on, strict cost (`cost_limit=2.0`, higher initial penalty, `pessimistic=True`).
- `osrp_vector_safedreamer_hallu_relaxed`: hallu Level2 on, relaxed safety (`cost_limit=8.0`, low initial penalty, `pessimistic=False`).
- Logdir base for both train/eval: `./logdir_phase2/` (eval-only writes to `./logdir_phase2_eval/`); final path is `<base>/<timestamp>_<method>_<task>_<seed>`.

## One-liner commands (short sanity runs)
Run from repo root. Override env vars `RUN_STEPS`/`LOGDIR_ROOT`/`EVAL_EVERY` as needed.

Local example (50k steps default):
```
bash scripts/run_phase2_baselines.sh clean 0 50000
bash scripts/run_phase2_baselines.sh hallu_strict 0 50000
bash scripts/run_phase2_baselines.sh hallu_relaxed 0 50000
bash scripts/run_phase2_baselines.sh unsafe 0 50000
```

Remote GPU example (via `r.sh`, conda env `safedreamer`):
```
cd ~/Codex/WM && ./r.sh "cd /root/autodl-tmp/projects/SafeDreamer && conda run -n safedreamer bash scripts/run_phase2_baselines.sh clean 0 20000"
cd ~/Codex/WM && ./r.sh "cd /root/autodl-tmp/projects/SafeDreamer && conda run -n safedreamer bash scripts/run_phase2_baselines.sh hallu_strict 0 20000"
cd ~/Codex/WM && ./r.sh "cd /root/autodl-tmp/projects/SafeDreamer && conda run -n safedreamer bash scripts/run_phase2_baselines.sh hallu_relaxed 0 20000"
cd ~/Codex/WM && ./r.sh "cd /root/autodl-tmp/projects/SafeDreamer && conda run -n safedreamer bash scripts/run_phase2_baselines.sh unsafe 0 20000"
```

## Eval-only helper
- Script: `scripts/eval_phase2_baselines.sh {baseline} <ckpt_dir_or_file> [seed] [eval_steps]`
- If you pass a logdir, it auto-picks the newest `checkpoint*.ckpt` (fallback `checkpoint.ckpt`).
- Example (remote, 5 eval episodes by default):
```
cd ~/Codex/WM && ./r.sh "cd /root/autodl-tmp/projects/SafeDreamer && conda run -n safedreamer bash scripts/eval_phase2_baselines.sh hallu_strict /root/autodl-tmp/projects/SafeDreamer/logdir_phase2/<run_dir> 0 2000"
```

## Quick sanity-check checklist
- After a short train, inspect `metrics.jsonl` in the run logdir (tail a few lines):
  - Clean: reward rises, cost near 0.
  - Hallu strict: reward stagnates ≈0, cost near 0 (frozen robot).
  - Hallu relaxed: reward > strict, cost noticeably higher.
  - Unsafe: reward highest early, cost highest.
- Videos: use existing demos (`demo_clean_v2.mp4`, `demo_hallu_level2_v2.mp4`) for visual reference if needed.
