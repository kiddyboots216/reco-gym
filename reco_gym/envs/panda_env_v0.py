# Omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#   
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# Beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action
import numpy as np
import gym

from numpy import array, diag, exp, matmul, mod
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from gym.spaces import Discrete, Dict, Box

from .session import OrganicSessions
from .context import DefaultContext
from .configuration import Configuration
from .observation import Observation

from .features.time import DefaultTimeGenerator

# Default arguments for toy environment ------------------------------------

# inherit most arguments from abstract class
env_args = {
    'max_time': 789,
    'num_products': 10,
    'num_users': 100,
    'random_seed': np.random.randint(2 ** 31 - 1),
    # Markov State Transition Probabilities.
    'prob_leave_bandit': 0.01,
    'prob_leave_organic': 0.01,
    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25,
}
env_1_args = {
    **env_args,
    **{
        'K': 5,
        'sigma_omega_initial': 1,
        'sigma_omega': 0.1,
        'number_of_flips': 0,
        'sigma_mu_organic': 3,
        'change_omega_for_bandits': False,
    }
}
panda_env_args = {
    **env_1_args,
    **{
        'penalty': False,
    }
}


# Maps behaviour into ctr - organic has real support ctr is on [0,1].
def ff(xx, aa = 5, bb = 2, cc = 0.3, dd = 2, ee = 6):
    # Magic numbers give a reasonable ctr of around 2%.
    return sigmoid(aa * sigmoid(bb * sigmoid(cc * xx) - dd) - ee)
# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2

# Environment definition.
# Amazon = Organic
# Facebook = Bandit
class PandaEnv0(gym.Env):
    def __init__(self):
        gym.Env.__init__(self)
        self.state = organic

    def init_gym(self, args):
        self.config = Configuration(args)

        # Defining Action Space.
        self.action_space = Discrete(self.config.num_products)

        # Defining Observation Space
        self.observation_space = Box(
            low = np.array([0, 0, 0]),
            high = np.array([self.config.max_time,
                            self.config.num_users,
                            self.config.num_products]),
            dtype = np.int32,
            )
        if 'time_generator' not in args:
            self.time_generator = DefaultTimeGenerator(self.config)
        else:
            self.time_generator = self.config.time_generator

        # Setting random seed for the first time.
        self.reset_random_seed()

        if 'agent' not in args:
            self.agent = None
        else:
            self.agent = self.config.agent

        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

        # Set random seed for second time, ensures multiple epochs possible.
        self.reset_random_seed()
        self.reset()

    def step(self, action):
        """
        Parameters
        ----------
        action_id : int between 1 and num_products indicating which
                 product recommended (aka which ad shown), or None if
                 user is on Amazon
        Returns
        -------
        observation, reward, done, info : tuple
            observation (tuple):
                time (0, max_time=789)
                user_id (0, num_users)
                product_id (0=Facebook, 1-num_products=Amazon products)
            reward (float):
                1 - user clicked on ad
                0 - user is on Amazon
                -1 - user didn't click on ad
            done (bool):
                time to reset
            info (dict):
                always an empty dict

        """
        assert self.state in (bandit, organic, stop)
        if self.state == organic:
            return self.step_organic()
        if self.state == bandit:
            return self.step_bandit(action)
        if self.state == stop:
            return self.step_stop()

    def step_organic(self):
        # look at a new product
        self.update_product_view()
        # user might close browser, go to FB or continue shopping
        self.update_state()
        # product-id on Amazon is in (1, num_products)
        obs = np.array((self.current_time, self.current_user_id, self.product_view))
        return obs, 0, True if self.state == stop else False, {}

    def step_bandit(self, action_id):
        rew = self.draw_click(action_id)
        if rew == 1:
            return self.step_successful_click()
        return self.step_unsuccessful_click()

    def step_successful_click(self):
        obs, _, _, _ = self.step_organic()
        return obs, 1, True if self.state == stop else False, {}

    def step_unsuccessful_click(self):
        # user didn't click ad but can still change state
        self.update_state()
        # product-id on FB is 0
        obs = np.array((self.current_time, self.current_user_id, 0))
        return obs, -1 if self.config.penalty else 0, True if self.state == stop else False, {}

    def step_stop(self):
        obs = np.array((self.current_time, self.current_time, 0))
        return obs, 0, True, {}

    def set_static_params(self):
        # Initialise the state transition matrix which is 3 by 3
        # high level transitions between organic, bandit and leave.
        self.state_transition = array([
            [0, self.config.prob_organic_to_bandit, self.config.prob_leave_organic],
            [self.config.prob_bandit_to_organic, 0, self.config.prob_leave_organic],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # Initialise Gamma for all products (Organic).
        self.Gamma = self.rng.normal(
            size = (self.config.num_products, self.config.K)
        )

        # Initialise mu_organic.
        self.mu_organic = self.rng.normal(
            0, self.config.sigma_mu_organic,
            size = (self.config.num_products)
        )

        # Initialise beta, mu_bandit for all products (Bandit).
        self.generate_beta(self.config.number_of_flips)

    def reset_random_seed(self, epoch = 0):
        # Initialize Random State.
        assert (self.config.random_seed is not None)
        self.rng = RandomState(self.config.random_seed + epoch)
    
    # Create a new user.
    def reset(self):
        self.first_step = True
        self.state = organic  # Manually set first state as Organic.

        self.time_generator.reset()
        if self.agent:
            self.agent.reset()

        self.current_time = self.time_generator.new_time()
        self.current_user_id = np.random.randint(self.config.num_users)

        # Record number of times each product seen for static policy calculation.
        self.organic_views = np.zeros(self.config.num_products)
        self.omega = self.rng.normal(
            0, self.config.sigma_omega_initial, size = (self.config.K, 1)
        )
        self.update_product_view()
        return [self.current_time, self.current_user_id, self.product_view]

    # Update user state to one of (organic, bandit, leave) and their omega (latent factor).
    def update_state(self):
        self.state = self.rng.choice(3, p = self.state_transition[self.state, :])
        assert (hasattr(self, 'time_generator'))
        old_time = self.current_time
        self.current_time = self.time_generator.new_time()
        time_delta = self.current_time - old_time
        omega_k = 1 if time_delta == 0 else time_delta

        # And update omega.
        if self.config.change_omega_for_bandits or self.state == organic:
            self.omega = self.rng.normal(
                self.omega,
                self.config.sigma_omega * omega_k, size = (self.config.K, 1)
            )

    # Sample a click as response to recommendation when user in bandit state
    # click ~ Bernoulli().
    def draw_click(self, recommendation):
        # Personalised CTR for every recommended product.
        ctr = ff(matmul(self.beta, self.omega)[:, 0] + self.mu_bandit)
        click = self.rng.choice(
            [0, 1],
            p = [1 - ctr[recommendation], ctr[recommendation]]
        )
        return click

    # Sample the next organic product view.
    def update_product_view(self):
        log_uprob = matmul(self.Gamma, self.omega)[:, 0] + self.mu_organic
        log_uprob = log_uprob - max(log_uprob)
        uprob = exp(log_uprob)
        self.product_view = int(
            self.rng.choice(
                self.config.num_products,
                p = uprob / sum(uprob)
            )
        )

    def generate_beta(self, number_of_flips):
        """Create Beta by flipping Gamma, but flips are between similar items only"""
        if number_of_flips == 0:
            self.beta = self.Gamma
            self.mu_bandit = self.mu_organic
            return
        P, K = self.Gamma.shape
        index = list(range(P))

        prod_cov = matmul(self.Gamma, self.Gamma.T)
        prod_cov = prod_cov - diag(
            diag(prod_cov))  # We are always most correlated with ourselves so remove the diagonal.

        prod_cov_flat = prod_cov.flatten()

        already_used = dict()
        flips = 0
        pcs = prod_cov_flat.argsort()[::-1]  # Find the most correlated entries
        for ii, jj in [(int(p / P), mod(p, P)) for p in pcs]:  # Convert flat indexes to 2d indexes
            # Do flips between the most correlated entries
            # provided neither the row or col were used before.
            if not (ii in already_used or jj in already_used):
                index[ii] = jj  # Do a flip.
                index[jj] = ii
                already_used[ii] = True  # Mark as dirty.
                already_used[jj] = True
                flips += 1

                if flips == number_of_flips:
                    self.beta = self.Gamma[index, :]
                    self.mu_bandit = self.mu_organic[index]
                    return

        self.beta = self.Gamma[index, :]
        self.mu_bandit = self.mu_organic[index]

if __name__ == "__main__":
    env = PandaEnv0()
    env.init_gym(env_1_args)
