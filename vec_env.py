"""
An interface for asynchronous vectorized environments.
"""

import ctypes
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Array, Process

import gym
import numpy as np
from baselines import logger

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}
_CT_TO_NP = {v: k for k, v in _NP_TO_CT.items()}


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        logger.warn('Render not defined for %s' % self)


class ShmemVecEnv(VecEnv):
    """
    An AsyncEnv that uses multiprocessing to run multiple
    environments in parallel.
    """

    def __init__(self, env_fns, spaces=None):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        if spaces:
            observation_space, action_space = spaces
        else:
            logger.log('Creating dummy env object to get spaces')
            with logger.scoped_configure(format_strs=[]):
                dummy = env_fns[0]()
                observation_space, action_space = dummy.observation_space, dummy.action_space
                dummy.close()
                del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        obs_spaces = observation_space.spaces if isinstance(self.observation_space, gym.spaces.Tuple) else (
            self.observation_space,)
        self.obs_bufs = [tuple(Array(_NP_TO_CT[s.dtype.type], int(np.prod(s.shape))) for s in obs_spaces) for _ in
                         env_fns]
        self.obs_shapes = [s.shape for s in obs_spaces]
        self.obs_dtypes = [s.dtype for s in obs_spaces]

        self.parent_pipes = []
        self.procs = []
        for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
            wrapped_fn = CloudpickleWrapper(env_fn)
            parent_pipe, child_pipe = Pipe()
            proc = Process(target=_subproc_worker,
                           args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes))
            proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
        self.waiting_step = False

    def reset(self):
        if self.waiting_step:
            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def _decode_obses(self, obs):
        """
        Turn the observation responses into a single numpy
        array, possibly via shared memory.
        """
        obs = []
        for i, shape in enumerate(self.obs_shapes):
            bufs = [b[i] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[i]).reshape(shape) for b in bufs]
            obs.append(np.array(o))
        return tuple(obs) if len(obs) > 1 else obs[0]


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_buf, obs_shape):
    """
    Control a single environment instance using IPC and
    shared memory.

    If obs_buf is not None, it is a shared-memory buffer
    for communicating observations.
    """

    def _write_obs(obs):
        if not isinstance(obs, tuple):
            obs = (obs,)
        for o, b, s in zip(obs, obs_buf, obs_shape):
            dst = b.get_obj()
            dst_np = np.frombuffer(dst, dtype=_CT_TO_NP[dst._type_]).reshape(s)  # pylint: disable=W0212
            np.copyto(dst_np, o)

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    finally:
        env.close()
