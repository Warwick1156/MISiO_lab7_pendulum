# Based on https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
# Adds noise for exploration purposes
import numpy as np

class Noise:
    def __init__(self, mean, std, theta=0.15, dt=1e-2, initial_noise=None):
        self.theta = theta
        self.mean = mean
        self.std = std
        self.dt = dt
        self.initial_noise = initial_noise

        # self.x_prev = None
        self.reset()

    def __call__(self):
        noise = (self.previous_noise + self.theta * (self.mean - self.previous_noise) * self.dt + self.std * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.previous_noise = noise
        return noise

    def reset(self):
        if self.initial_noise is not None:
            self.previous_noise = self.initial_noise
        else:
            self.previous_noise = np.zeros_like(self.mean)

