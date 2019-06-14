# Omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#   
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# Beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action
import numpy as np
from numpy import array, diag, exp, matmul, mod
from scipy.special import expit as sigmoid
from .abstract import AbstractEnv, env_args, organic
from .reco_env_v1 import RecoEnv1

# Default arguments for toy environment ------------------------------------

# inherit most arguments from abstract class
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
class PandaEnv0(RecoEnv1):
    def __init__(self):
        super().__init__()

    # @override
    def step(self, action_id):
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
        info = {}
        # assume we are in Amazon, if not this will get updated
        rew = 0
        assert self.state in (bandit, organic, stop)
        if self.state == bandit:
            # print("We're on FB")
            # show an ad and see whether user clicks it
            rew = self.draw_click(action_id)
            if rew == 1:
                # user clicked the ad, proceed to Amazon
                self.state = organic
            else:
                rew = -1
                # user didn't click ad but can still change state
                self.update_state()
                done = True if self.state == stop else False
                # product-id on FB is 0
                obs = np.array((self.current_time, self.current_user_id, 0))
                return obs, rew, done, info
        if self.state == organic:
            # print("We're on Amazon")
            # look at a new product
            self.update_product_view()
            # user might close browser, go to FB or continue shopping
            self.update_state()
            # product-id on Amazon is in (1, num_products)
            obs = np.array((self.current_time, self.current_user_id, self.product_view))
            done = True if self.state == stop else False
            return obs, rew, done, info
        if self.state == stop:
            obs = np.array((self.current_time, self.current_time, 0))
            done = True
            return obs, rew, done, info