import functools
from typing import Optional, Union, Tuple, Any

import jax
import jax.numpy as jnp
import chex


from gymnax.wrappers.purerl import GymnaxWrapper
from gymnax.environments import environment, spaces

import seaborn as sns

from typing import Any, Dict, Tuple
import jax
from jax import Array
from flax import struct
from gymnax.environments.environment import (
    Environment as GymnaxEnv,
    EnvParams,
    EnvState,
)
from gymnax.environments.spaces import Discrete as GymnaxDiscrete, Box as GymnaxBox


from functools import partial

class DreamerWrapper(GymnaxWrapper):

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, Any]:  # dict]:
        if params is None:
            params = self.default_params
        # key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        # obs_re, state_re = self.reset_env(key_reset, params)
        
        # Auto-reset environment based on termination
        # state = jax.tree_map(
        #     lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        # )
        # obs = jax.lax.select(done, obs_re, obs_st)
        return obs_st, state_st, reward, done, info

def render_minatar(obs, colors):
    n_channels = obs.shape[-1]
    numerical_state = (
        jnp.amax(obs * jnp.reshape(jnp.arange(n_channels) + 1, (1, 1, -1)), 2)
    ).astype(jnp.int32)

    new_obs = colors[numerical_state]

    return new_obs


class MinAtarPixel(GymnaxWrapper):

    def __init__(self, env):
        super().__init__(env)
        n_channels = env.obs_shape[-1]
        cmap = sns.color_palette("cubehelix", n_channels)
        cmap.insert(0, (0, 0, 0))
        self.colors = jnp.array(list(cmap))
        self.obs_shape = (*env.obs_shape[:-1], 3) #RGB

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, Any]:
        
        obs, state, reward, done, info = self._env.step(
            key,
            state,
            action,
            params
        )

        obs = render_minatar(obs, self.colors)
        return obs, state, reward, done, info
     
    def reset(
            self,
            key: chex.PRNGKey,
            params: Optional[environment.EnvParams] = None
    )-> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        return render_minatar(obs, self.colors), state
    
    def observation_space(self, params: environment.EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)
    
class PixelNoise(GymnaxWrapper):

    def __init__(self, env, noise_sigma, **kwargs):
        super().__init__(env)
        self.noise_sigma = noise_sigma

    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, Any]:
        key_step, key_noise = jax.random.split(key)
        obs, state, reward, done, info = self._env.step(
            key_step,
            state,
            action,
            params
        )

        obs = obs + jax.random.normal(key_noise, obs.shape) * self.noise_sigma
        obs = jnp.clip(obs, 0., 1.)
        return obs, state, reward, done, info
     
    def reset(
            self,
            key: chex.PRNGKey,
            params: Optional[environment.EnvParams] = None
    )-> Tuple[chex.Array, environment.EnvState]:
        key, key_noise = jax.random.split(key)
        obs, state = self._env.reset(key, params)
        obs = obs + jax.random.normal(key_noise, obs.shape) * self.noise_sigma
        obs = jnp.clip(obs, 0., 1.)
        return obs, state

    def observation_space(self, params: environment.EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    
@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0., 0, 0., 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        # info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info

    def state_space(self, state):
        return self._env.state_space(state.env_state)
