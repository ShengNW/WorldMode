import re
import json
import os

import embodied
import numpy as np
import jax

class CostEma:

  def __init__(self, initial=0):
    self.value = initial

class Arrive:

  def __init__(self):
    self.value = []

def train_eval(
    agent, train_env, eval_env, train_replay, eval_replay, logger, args, lag):
  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  diag_path = logdir / 'diag.jsonl'

  def _to_py(x):
    if isinstance(x, (float, int, np.floating, np.integer)):
      return float(x)
    if hasattr(x, 'tolist'):
      arr = x.tolist()
      try:
        return float(arr)
      except Exception:
        return arr
    return x

  def _make_diag(step_value, metrics_dict):
    # Accept both raw mets (no prefix) and aggregated metrics (with train/ prefix).
    def _get(key, default=0.0):
      if key in metrics_dict:
        return metrics_dict.get(key, default)
      pref = f'train/{key}'
      return metrics_dict.get(pref, default)

    # Pull out lag info if present.
    lag_keys = [
        'lag_Jc_hard', 'lag_Jc_ghost',
        'lag_limit_hard', 'lag_limit_ghost',
        'lag_g_hard', 'lag_g_ghost',
        'lag_lambda_lr_hard', 'lag_lambda_lr_ghost',
        'lag_upper_bound_hard', 'lag_upper_bound_ghost',
        'lag_lambda_pre_hard', 'lag_lambda_pre_ghost',
        'lag_lambda_post_hard', 'lag_lambda_post_ghost',
        'lag_penalty_pre_hard', 'lag_penalty_pre_ghost',
        'lag_penalty_post_hard', 'lag_penalty_post_ghost',
        'lag_penalty_hard', 'lag_penalty_ghost',
        'lag_update_calls',
    ]
    lag_info = {k: _get(k, None) for k in lag_keys if _get(k, None) is not None}

    # Device transfer to host to avoid stale device buffers.
    metrics_local = jax.device_get({k: v for k, v in metrics_dict.items()})
    lag_local = jax.device_get(lag_info) if lag_info else {}

    rec = {
        'step': int(step_value),
        'cwd': os.getcwd(),
        'logdir': str(os.path.abspath(str(logdir))),
    }
    # Core multipliers and penalties.
    rec['nu_hard'] = _to_py(_get('lagrange_multiplier', 0.0))
    rec['nu_ghost'] = _to_py(_get('ghost_lagrange_multiplier', 0.0))
    rec['penalty_hard'] = _to_py(_get('penalty', 0.0))
    rec['penalty_ghost'] = _to_py(_get('ghost_penalty', 0.0))
    rec['ghost_usage_mean'] = _to_py(_get('ghost_usage_mean', 0.0))
    rec['ghost_label_mean'] = _to_py(_get('ghost_label_mean', 0.0))
    rec['ghost_pred_mean'] = _to_py(_get('ghost_pred_mean', 0.0))
    rec['grad_overflow'] = _to_py(_get('wm_grad_overflow', _get('actor_grad_overflow', 0.0)))
    # Dual detailed diagnostics if present.
    for key, val in lag_local.items():
      rec[key] = _to_py(val)
    # Safety net: capture any lag* keys that slipped through with other prefixes.
    for key, val in metrics_local.items():
      if 'lag' in key:
        rec[key] = _to_py(val)
    rec['lag_keys_present'] = [k for k in metrics_dict.keys() if 'lag' in k]
    rec['keys_sample'] = list(metrics_dict.keys())[:20]
    return rec

  def _write_diag(rec):
    diag_path.parent.mkdirs()
    with open(diag_path, 'a') as f:
      f.write(json.dumps(rec) + '\n')
      f.flush()
      os.fsync(f.fileno())
  should_expl = embodied.when.Until(args.expl_until)
  train_ratio_fn = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_train = lambda step: int(max(1, train_ratio_fn(step)))
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  step = logger.step
  cost_ema = CostEma(0.0)
  train_arrive_num = Arrive()
  eval_arrive_num = Arrive()
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length,
        'score': score,
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    print(f'Episode has {length} steps and return {score:.1f}.')
    if 'cost' in ep.keys():
      cost = float(ep['cost'].astype(np.float64).sum())
      logger.add({
          'cost': cost,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      print(f'Episode has {length} steps and  cost {cost:.1f}.')
      # lag.add_cost(cost)
      cost_ema.value = cost_ema.value * 0.99 + cost * 0.01
      logger.add({
          'cost_ema': cost_ema.value,
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
      if step > 5000:
        lag.pid_update(cost_ema.value, step)
    if 'arrive_dest' in ep.keys():
      if mode == 'train':
        train_arrive_num.value.append(int(ep['arrive_dest'][-1]))
        if len(train_arrive_num.value) == 10:
          arrive_rate = sum(train_arrive_num.value) /  10
          train_arrive_num.value = []
          logger.add({
              'arrive_rate': arrive_rate,
          }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
          print(f'train 10 episodes has average arrive rate {arrive_rate:.2f}.')

      else:
        eval_arrive_num.value.append(int(ep['arrive_dest'][-1]))
        if len(eval_arrive_num.value) == 10:
          arrive_rate = sum(eval_arrive_num.value) /  10
          eval_arrive_num.value = []
          logger.add({
              'arrive_rate': arrive_rate,
          }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
          print(f'eval 10 episodes has average arrive rate {arrive_rate:.2f}.')


    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix=f'{mode}_stats')

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)
  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
  logger.add(metrics.result())
  logger.write()
  _write_diag(_make_diag(step, {}))

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      current = metrics.result()
      logger.add(current)
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
      diag_source = current
      _write_diag(_make_diag(step, diag_source))
  driver_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps), lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
    driver_train(policy_train, steps=100, lag=lag.lagrange_penalty, lag_p=lag.delta_p, lag_i=lag.pid_i, lag_d=lag.pid_d)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
