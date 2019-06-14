from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import ray
import gym
# import reco_gym
# from reco_gym import env_1_args, Configuration
from ray.tune import run
from ray.tune.registry import register_env

env_name = "reco"

def env_creator(config):
    import reco_gym
    from reco_gym import env_1_args, Configuration
    # env_0_args is a dictionary of default parameters (i.e. number of products)

    # You can overwrite environment arguments here:
    env_1_args['random_seed'] = 42

    # Initialize the gym for the first time by calling .make() and .init_gym()
    env = gym.make('panda-gym-v0')
    env.init_gym(env_1_args)
    # env.reset()
    return env

# env_creator("boo")
if __name__ == "__main__":
    register_env(env_name, env_creator)
    ray.init(ignore_reinit_error=True)
    run(
        "PPO",
        name="RecoGymPPO",
    #     env="reco-gym-v1",
        config = {
            "lambda": 0.95, #0.5 to 0.99
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.1,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01, #0 to 0.2
            "train_batch_size": 500,
            "sample_batch_size": 10,
            "sgd_minibatch_size": 50,
            "num_sgd_iter": 10,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "vf_share_layers": True,
            "num_gpus": 0,
            "env": env_name,
            # "env": multienv_name,
            # These params are tuned from a fixed starting value.
            # "lambda": 0.95,
            # "clip_param": 0.2,
            "lr": 1e-4, #1e-2 to 1e-5
            # These params start off randomly drawn from a set.
        },
        )