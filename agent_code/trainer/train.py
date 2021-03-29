import pickle
import numpy as np
import random
from collections import namedtuple, deque
from typing import List
import events as e
import os

import sys
sys.path.append('../')

from agent_code.agent_SUN.callbacks import SAFE, ACTIONS, Q_save, Q_tracker, vicinity_set,vicinity_type,state_to_index,\
    initialize_Q, infer_local_features, state_to_features,features_to_stateIndex

from agent_code.agent_SUN.train import alpha, gamma, pacifier, TRANSITION_HISTORY_SIZE, \
    game_events_occurred, end_of_round, ESCAPED #reward_from_events


def setup_training(self):

    self.vicinity_type = vicinity_type
    self.vicinity_set = vicinity_set
    self.state_to_index = state_to_index

    if self.train or not os.path.isfile(Q_save):
        self.logger.info("Initiate Q-table for training.")
        self.Q = initialize_Q()
    else:
        self.logger.info("Loading saved Q-table from previous training.")
        with open(Q_save, "rb") as file:
            self.Q = pickle.load(file)

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.eventList = []
    self.rewardList = []
    self.alpha = alpha
    self.gamma = gamma
    self.Q_tracker = np.zeros(self.Q.shape)
    self.pacifier = pacifier

    return self


# NEW - without movement nudge

def reward_from_events(self, events: List[str]) -> int:
    """    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage certain behavior. """

    # this is the immediate reward accredited to last action
    game_rewards_immediate = {
        # penalties
        e.GOT_KILLED: -100, #-100
        e.INVALID_ACTION: -0.01,
        # incentives
        e.SURVIVED_ROUND: 5,
        e.COIN_COLLECTED: 1,
        e.BOMB_DROPPED: 0.02,
        # MOVING_TOWARDS_TARGET: 0.005, removed from further training otherwise encouraging just moving around
        # MOVING_AWAY_TARGET: -0.005,
    }

    # this is the reward to be credited to previous action, esp. bomb dropped 4 steps ago
    game_rewards_b4 = {
        # penalties
        e.KILLED_SELF: -100, #-100
        # incentives
        e.BOMB_EXPLODED: 0.1,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 0.1,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 5,
    }

    game_rewards_p4 = {
        ESCAPED: 0.25, #-10 reward for each step in the path of last 4 steps away from bombing - result is to drop bombs without collecting coins
        e.KILLED_SELF: -0.25, #-10

    }

    reward_sum = 0
    reward_sum_b4 = 0
    reward_sum_p4 = 0
    for event in events:
        if event in game_rewards_immediate:
            reward_sum += game_rewards_immediate[event]
        if event in game_rewards_b4:
            reward_sum_b4 += game_rewards_b4[event]
        if event in game_rewards_p4:
            reward_sum_p4 += game_rewards_p4[event]

    self.logger.info(f"Awarded {reward_sum} (current) and {reward_sum_b4} (previous) for events {', '.join(events)}")

    return reward_sum, reward_sum_b4, reward_sum_p4


### OLD - same as agent
# def reward_from_events(self, events: List[str]) -> int:
#     """    *This is not a required function, but an idea to structure your code.*
#     Here you can modify the rewards your agent get so as to en/discourage certain behavior. """
#
#     # this is the immediate reward accredited to last action
#     game_rewards_immediate = {
#         # penalties
#         e.GOT_KILLED: -100,
#         e.INVALID_ACTION: -0.01,
#         # incentives
#         e.SURVIVED_ROUND: 5,
#         e.COIN_COLLECTED: 1,
#         e.BOMB_DROPPED: 0.02,
#         # removed after team discussion - should not encourage unnecessary movements
#         # e.MOVED_UP:0.01,
#         # e.MOVED_RIGHT:0.01,
#         # e.MOVED_DOWN:0.01,
#         # e.MOVED_LEFT:0.01,
#         MOVING_TOWARDS_TARGET: 0.005,
#         MOVING_AWAY_TARGET: -0.005,
#     }
#
#     # this is the reward to be credited to previous action, esp. bomb dropped 4 steps ago
#     game_rewards_b4 = {
#         # penalties
#         e.KILLED_SELF: -100,
#         # incentives
#         e.BOMB_EXPLODED: 0.1,
#         e.CRATE_DESTROYED: 0.1,
#         e.COIN_FOUND: 0.1,
#         e.KILLED_OPPONENT: 5,
#         e.SURVIVED_ROUND: 5,
#     }
#
#     game_rewards_p4 = {
#         ESCAPED: 10, #reward for each step in the path of last 4 steps away from bombing
#         e.KILLED_SELF: -10,
#
#     }
#
#     reward_sum = 0
#     reward_sum_b4 = 0
#     reward_sum_p4 = 0
#     for event in events:
#         if event in game_rewards_immediate:
#             reward_sum += game_rewards_immediate[event]
#         if event in game_rewards_b4:
#             reward_sum_b4 += game_rewards_b4[event]
#         if event in game_rewards_p4:
#             reward_sum_p4 += game_rewards_p4[event]
#
#     self.logger.info(f"Awarded {reward_sum} (current) and {reward_sum_b4} (previous) for events {', '.join(events)}")
#
#     return reward_sum, reward_sum_b4, reward_sum_p4
#
