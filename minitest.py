import gym, reco_gym
import numpy as np

# env_0_args is a dictionary of default parameters (i.e. number of products)
from reco_gym import env_1_args, Configuration

# You can overwrite environment arguments here:
env_1_args['random_seed'] = 42

# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('panda-gym-v0')
env.init_gym(env_1_args)
i = 0
done = False
while i < 1000:
    while not done:
        action = np.random.randint(10)
    #     from IPython.core.debugger import set_trace; set_trace()

        observation, reward, done, info = env.step(action)
        assert observation is not None
        print(f"Step: {i} - Action: {action} - Observation: {observation} - Reward: {reward} - Done: {done}")
        i += 1
    env.reset()
    done = False
#     print(i)