"""RL layer: Gymnasium environment, action/state encodings, reward; agents from Step 6 onwards.

Importing this package registers ``ThermoOpt-DataCenter-v0``:

    import gymnasium as gym
    import backend.rl  # noqa: F401  (registers the environment)
    env = gym.make("ThermoOpt-DataCenter-v0", episode_minutes=60)
    env = gym.make("ThermoOpt-DataCenter-v0", representation="tabular")
"""

from gymnasium.envs.registration import register, registry

ENV_ID = "ThermoOpt-DataCenter-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="backend.rl.environment:DataCenterEnv")
