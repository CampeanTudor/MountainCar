
import numpy as np


class Analytics:

    def __init__(self):
        self.rewards = []
        self.episodes = []
        self.number_of_steps = []
        self.time_for_episodes = []
        self.episodes_before_solve = None

    def add_info(self, reward=None, episode_number=None, number_of_steps=None, time_for_episode=None,
                 episodes_before_solve=None):

        if reward is not None:
            self.rewards.append(reward)
        if episode_number is not None:
            self.episodes.append(episode_number)
        if number_of_steps is not None:
            self.number_of_steps.append(number_of_steps)
        if time_for_episode is not None:
            self.time_for_episodes.append(time_for_episode)
        if episodes_before_solve is not None:
            self.episodes_before_solve = episodes_before_solve