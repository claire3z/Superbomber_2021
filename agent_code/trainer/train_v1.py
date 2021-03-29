import pickle
import numpy as np
import random
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import infer_local_features, state_to_features,features_to_stateIndex, ACTIONS, Q_save, path

# learning rate -> 0: agent learns nothing (exclusively exploiting prior knowledge), 1: agent considers only the most recent information (ignoring prior knowledge to explore possibilities).
alpha = 0.5
# discount rate -> 0: agent only considers current rewards in update, 1: agent striving for a long-term high reward.
gamma = 0.5
# rationale: balanced between current and next state

pacifier = True # if True, no BOMB -> WAIT, to avoid killing itself too quickly at initial training
Q_tracker = path+'Q_tracker_SUN.pk' # initialize once and keep tracking of update stats in Q-table

# This is only an example!
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

# Hyper parameters
TRANSITION_HISTORY_SIZE = 5  # keep last 5 transitions in order to redistribute reward correctly and update Q-table post-effect
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Define customized events and rewards
# PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """ Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values. """
    # Example: Setup an array that will note transition tuples (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.eventList = []
    self.rewardList = []
    self.alpha = alpha
    self.gamma = gamma
    self.Q_tracker = np.zeros(self.Q.shape)
    self.pacifier = pacifier  # if True, no BOMB -> WAIT; rational: prevent killing itself too quickly initial training
    return self


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """ Called once per step to allow intermediate rewards based on game events.
    When this method is called, self.events will contain a list of all game events relevant to your agent that occurred during the previous step.
    Consult settings.py to see what events are tracked. You can hand out rewards to your agent based on these events and your knowledge of the (new) game state.
    This is *one* of the places where you could update your agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`"""

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # print(f'Events: {", ".join(map(repr, events))}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER)

    old_features,moves = state_to_features(self,old_game_state)
    old_state = features_to_stateIndex(self,old_features)

    new_features,moves = state_to_features(self, new_game_state)
    new_state = features_to_stateIndex(self, new_features)

    # NEW
    if self_action is not None:
        action = ACTIONS.index(self_action)
        reward, reward_b4 = reward_from_events(self, events)
        # print(f"agent = {new_game_state['self']}, reward = {reward} (current) and {reward_b4} (previous) for events {', '.join(events)},")

        self.transitions.append(Transition(old_state,action,new_state,reward)) # append current reward
        # update Q-table with current reward
        self.Q[old_state, action] = (1 - self.alpha) * self.Q[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))
        self.Q_tracker[old_state, action] += 1
        # print(f"reward = {reward}, Q_table at [{old_state}, {action}] updated from {self.Q[old_state, action]} to {(1 - self.alpha) * self.Q[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))}")

        if reward_b4 != 0:
            # update t-4 in Q-table which is affected by the reward adjustment
            old_state, action, new_state, reward = self.transitions[0] # update t-4, due to modified reward
            reward = reward_b4
            self.Q[old_state, action] = (1 - self.alpha) * self.Q[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))
            self.Q_tracker[old_state, action] += 1
            # print(f"reward = {reward}, Q_table at [{old_state}, {action}] updated from {self.Q[old_state, action]} to {(1 - self.alpha) * self.Q[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))}")

    # OLD - reward always credited to last action
    # if self_action is not None and reward != 0:
    #     action = ACTIONS.index(self_action)
    #     self.Q[old_state, action] = (1 - self.alpha) * self.Q[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))
    #     print(f"reward = {reward}, Q_table at [{old_state}, {action}] updated from {self.Q[old_state, action]} to {(1 - self.alpha) * self.Q[old_state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))}")

    # There are some 'invalid_action' events that should not occur... checking maps --> if another agent step over first then invalid for my agent.
    if 'INVALID_ACTION' in events:
        print('\nStep:',old_game_state['step'],'->',new_game_state['step'])
        print('State:',self.transitions[-1][0],infer_local_features(self,old_game_state['self']))
        print('Action:', ACTIONS[self.transitions[-1][1]])
        print('Events:',events)
        print('Reward:', self.transitions[-1][3])
        print('Global_map:')
        print(self.global_map)
        print('Treasure_map:')
        print(self.treasure_map)

        print('>Self:', old_game_state['self'],'->',new_game_state['self'])
        print('>Coins:', old_game_state['coins'],'->',new_game_state['coins'])
        print('>Bombs:', old_game_state['bombs'],'->',new_game_state['bombs'])
        print('>Others:', old_game_state['others'],'->',new_game_state['others'])





def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game or when the agent died to hand out final rewards.
    This is similar to reward_update. self.events will contain all events that occurred during your agent's final step.
    This is *one* of the places where you could update your agent. This is also a good place to store an agent that you updated.
    :param self: The same object that is passed to all of your callbacks."""
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    game_events_occurred(self, last_game_state, last_action, last_game_state, events)
    print(f'<<< END >>>')
    print(f"self.Q_tracker: max={self.Q_tracker.max()}, sum={self.Q_tracker.sum()}, distribution: {np.unique(self.Q_tracker,return_counts=True)}")

    # Store the Q-table
    with open(Q_save, "wb") as file:
        pickle.dump(self.Q, file)

    with open(Q_tracker, "wb") as file:
        pickle.dump(self.Q_tracker, file)
    # np.savetxt('Q_tracker_SUN.csv', self.Q_tracker, delimiter=',') # for visualization


def reward_from_events(self, events: List[str]) -> int:
    """    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage certain behavior. """

    # this is the immediate reward accredited to last action
    game_rewards_immediate = {
        # penalties
        e.GOT_KILLED: -100,
        e.INVALID_ACTION: -1,
        # incentives
        e.SURVIVED_ROUND: 5,
        e.COIN_COLLECTED: 1,
        e.BOMB_DROPPED: 0.2,
        e.MOVED_UP:0.1,
        e.MOVED_RIGHT:0.1,
        e.MOVED_DOWN:0.1,
        e.MOVED_LEFT:0.1,
    }

    # this is the reward to be credited to previous action, esp. bomb dropped 4 steps ago
    game_rewards_b4 = {
        # penalties
        e.KILLED_SELF: -100,
        # incentives
        e.BOMB_EXPLODED: 0.1,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 0.1,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 5,
        # PLACEHOLDER: 5,
    }

    reward_sum = 0
    reward_sum_b4 = 0
    for event in events:
        if event in game_rewards_immediate:
            reward_sum += game_rewards_immediate[event]
        if event in game_rewards_b4:
            reward_sum_b4 += game_rewards_b4[event]
    self.logger.info(f"Awarded {reward_sum} (current) and {reward_sum_b4} (previous) for events {', '.join(events)}")

    return reward_sum, reward_sum_b4
