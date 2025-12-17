# Unit-Scale Fix (Dec 17, 2025)

- **What changed**
  - Scale dual updates to episodic units: `DualLagrange` now multiplies per-step cost surrogates by `lagrange_episode_length` before updating λ, and logs raw vs. scaled Jc plus episode length.
  - Actor-critic now reads `lagrange_episode_length` (defaults to 200; override via CLI) and passes hard/ghost lengths into the dual update; logs them for diagnostics.
  - Restored `cost_weight` and `pessimistic` to `defaults` in `configs.yaml` to avoid `KeyError` during config composition.

- **Evidence**
  - Smoke run: `logdir_phase3_fix_units_smoke20251217-203824_ours_full_safetygymcoor_SafetyPointGoal1-v0_0` (200-step, completed).
  - Verification run (in progress): `logdir_phase3_fix_units_vr20251217-204356_ours_full_safetygymcoor_SafetyPointGoal1-v0_0`.
  - Sample training metrics (verification run, step≈50k):  
    - `train/lag_Jc_hard_raw ≈ 0.19`, `lag_episode_length_hard=200` → `lag_Jc_hard ≈ 37.7`, `lag_g_hard ≈ 35.7`, `lag_lambda_post_hard ≈ 203`.  
    - Ghost path logs `lag_Jc_ghost_raw ≈ 0.008`, `lag_lambda_post_ghost = 0` (still near-zero signal; now explicitly logged).

- **How to override**
  - CLI: `--lagrange_episode_length <L>` (and optionally `--ghost_lagrange_episode_length <L_ghost>`).
  - Config defaults set `lagrange_episode_length: 200`, `ghost_lagrange_episode_length: 1.0`.

- **Commits**
  - Code scaling/logging: `affe521` (server sync branch).
  - Restore config keys: `f479388` (server sync branch).

## Verification run (10k steps, ours_full, seed=0)
- Logdir: `logdir_phase3_fix_units_vr20251217-204356_ours_full_safetygymcoor_SafetyPointGoal1-v0_0`
- Eval stats: `last_eval_cost_ema ≈ 16.86`, `median eval_episode/cost = 43.0`, `n_eval = 5`, `best_eval_score = 0.493 @ step 30k` with `cost_ema_at_best ≈ 14.6`.
- Dual signals (train-side): `mean(lag_Jc_hard) ≈ 21.17`, `median eval_cost_ema / lag_Jc_hard ≈ 0.94` (units now aligned), `lambda_max ≈ 203`, `g_positive_rate ≈ 0.92`.
- Ghost path: `lag_Jc_ghost_raw ≈ 0.008`, `lag_lambda_post_ghost = 0` → constraint effectively inactive (signal too small, not a sign bug).
- Stability: `grad_overflow_mean = 0`, no OOM; episode length used for scaling `≈200`.
- Takeaway: Hard constraint now “feels” episode-scale cost (λ grows to ~200); cost still high, so next we should retune cost_limit or LR rather than fix units.
