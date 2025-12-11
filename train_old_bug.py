import os
import importlib
import pathlib
import sys
import warnings
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers
import datetime


def main(argv=None):
  from . import agent as agt

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  now_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir_algo = config.logdir + now_time + '_' + str(config.method) + '_' + str(config.task) + '_' + str(config.seed)
  args = embodied.Config(
      **config.run, logdir=logdir_algo, use_cost=config.use_cost,
      batch_steps=config.batch_size * config.batch_length)
  print(config)
  os.environ['CUDA_VISIBLE_DEVICES'] = str(config.jax.logical_gpus)

  logdir = embodied.Path(logdir_algo)
  logdir.mkdirs()
  config.save(logdir / 'config.yaml')
  step = embodied.Counter()
  logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  lag = PIDLagrangian(config)
  try:

    if args.script == 'train':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_save':
      replay = make_replay(config, logdir / 'replay')
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_save(agent, env, replay, logger, args, lag)

    elif args.script == 'train_eval':
      replay = make_replay(config, logdir / 'replay')
      eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      eval_env = make_envs(config)  # mode='eval'
      cleanup += [env, eval_env]
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args, lag)

    elif args.script == 'train_holdout':
      replay = make_replay(config, logdir / 'replay')
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
        eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
      env = make_envs(config)
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      embodied.run.train_holdout(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'eval_only':
      env = make_envs(config, mode='eval')
      cleanup.append(env)
      agent = agt.Agent(env.obs_space, env.act_space, step, config)
      embodied.run.eval_only(agent, env, logger, args, lag)
      # Save a quick rollout video from the first env for visual sanity.
      try:
        import imageio
        frames = []
        env0 = env.envs[0] if hasattr(env, 'envs') else env
        env0.reset()
        for _ in range(int(config.run.steps)):
          # Sample random actions to force movement; avoids depending on agent state here.
          action = {k: v.sample() for k, v in env0.act_space.items() if k != 'reset'}
          obs = env0.step({**action, 'reset': False})
          frame = None
          if isinstance(obs, dict):
            frame = obs.get('image') or obs.get('image_far')
          if frame is not None:
            frames.append(frame)
        if frames:
          video_path = logdir / 'eval_render.mp4'
          imageio.mimwrite(video_path, frames, fps=10)
          print(f'[eval_only] saved video to {video_path}')
      except Exception as e:
        print(f'[eval_only] video dump failed: {e}')

    elif args.script == 'parallel':
      assert config.run.actor_batch <= config.envs.amount, (
          config.run.actor_batch, config.envs.amount)
      step = embodied.Counter()
      env = make_env(config)
      agent = agt.Agent(env.obs_space, env.act_space, step, config, lag)
      env.close()
      replay = make_replay(config, logdir / 'replay', rate_limit=True)
      embodied.run.parallel(
          agent, replay, logger, bind(make_env, config),
          num_envs=config.envs.amount, args=args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()
def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score|episode/cost'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ], multiplier)
  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    kw = {'online': config.replay_online}
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def make_envs(config, **overrides):
  suite, task = config.task.split('_', 1)
  ctors = []
  for index in range(config.envs.amount):
    ctor = lambda: make_env(config, **overrides)
    if config.envs.parallel != 'none':
      ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
      ctor = bind(wrappers.RestartOnException, ctor)
    ctors.append(ctor)
  envs = [ctor() for ctor in ctors]
  return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
  # You can add custom environments by creating and returning the environment
  # instance here. Environments with different interfaces can be converted
  # using `embodied.env}}6(|(((}(M(M(}M5(}4((5((51(5(19(AA(mt(((}(((((((}](](}(()}((}((((=!((((9(M((Q1((M(}(((()()()A%1(A%1((M1A%((((QA%11(A%}1A%(1(((}QA%(}QA%(}QA%(}}Q(}}}}Q}(}}}}Q}(}](}](}Q(}}Q(}Q((I(QIMI1A%15()!QA(UI0A%1|((((}}}|(((9(%A%1(}}(}}(}}(}}}}(}}}}}}}}(}}}}}}}}(}}}(}}}(}}}(}}}}(}}mt}}}(}}(}}(}}(}}}(}}}(}}}}}(}}}}}(}}}(}m}}}t(}m}}}}}}((}(Q(}}(((}(Q(}}((}(Q(}}(((}(Q(}}(((}9(UA%((A%((((q}q}-}}-}q}-}q}q((}(-}1}1}A%q((((}}Q((}}(}}}(}((}}}}mt(((}}}}}((}}}}}(((}}}}}(}}}}}}(}}(}}}}(}}}}}}(}}}(}}}(}}}}}}(}}}(}}}}}(}}}}}}lt(}}}}}}}}}}}(}}}(}}(}}}}(}}}}(}}}}}}(}}}}(}}}}}}(}}}}}}}}(()}}}|}}}|((