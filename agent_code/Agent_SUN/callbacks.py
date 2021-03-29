import os
import pickle
import random
from collections import namedtuple,deque
import numpy as np
import settings as s
import time

SAFE = 99
TRACKER_HISTORY = 20
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
Q_save = 'model/Q_SUN.pk'

eps_train = 0.15
eps = 0.1
# pacifier = True

####### for record keeping and experiment analysis REMOVE BEFORE COMPETITION TODO
path = 'model/game/alone/eps_0.5/'
version = '_50000_eps_0.5' #got killed -100
Q_save = path+'Q'+version+'.pk'
Q_tracker = path+'Q'+version+'_tracker.pk' # initialize once and keep tracking of update stats in Q-table


#######

Vicinity = namedtuple('Vicinity', ['neighbours', 'explosion_range', 'actions'])
# neighbours list of tuples [(dx,dy),...] relative location of neighbour if exists
# explosion range [x.min, x.max,y.min, y.max],
# actions: only allowed if neighbour exists in that direction, a subset of ['UP', 'RIGHT', 'DOWN', 'LEFT'] + ['WAIT', 'BOMB']
v0 = Vicinity([(0,-1),(1,0),(0,1),(-1,0)],[-s.BOMB_POWER,s.BOMB_POWER,-s.BOMB_POWER,s.BOMB_POWER],[0,1,2,3,4,5])
v1 = Vicinity([(0,-1),(0,1)],[0,0,-s.BOMB_POWER,s.BOMB_POWER],[0,2,4,5])
v2 = Vicinity([(1,0),(-1,0)],[-s.BOMB_POWER,s.BOMB_POWER,0,0],[1,3,4,5])
v3 = Vicinity([(0,-1),(1,0),(0,1)],[0,s.BOMB_POWER,-s.BOMB_POWER,s.BOMB_POWER],[0,1,2,4,5])
v4 = Vicinity([(0,-1),(0,1),(-1,0)],[-s.BOMB_POWER,0,-s.BOMB_POWER,s.BOMB_POWER],[0,2,3,4,5])
v5 = Vicinity([(1,0),(0,1),(-1,0)],[-s.BOMB_POWER,s.BOMB_POWER,0,s.BOMB_POWER],[1,2,3,4,5])
v6 = Vicinity([(0,-1),(1,0),(-1,0)],[-s.BOMB_POWER,s.BOMB_POWER,-s.BOMB_POWER,0],[0,1,3,4,5])
v7 = Vicinity([(1,0),(0,1)],[0,s.BOMB_POWER,0,s.BOMB_POWER],[1,2,4,5])
v8 = Vicinity([(0,-1),(1,0)],[0,s.BOMB_POWER,-s.BOMB_POWER,0],[0,1,4,5])
v9 = Vicinity([(0,-1),(-1,0)],[-s.BOMB_POWER,0,-s.BOMB_POWER,0],[0,3,4,5])
v10 = Vicinity([(0,1),(-1,0)],[-s.BOMB_POWER,0,0,s.BOMB_POWER],[2,3,4,5])
vicinity_set =(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10)






# This is a pre-generated file based on excel
vicinity_type = np.loadtxt('./agent_code/agent_SUN/model/vicinity_types.csv', delimiter=',')

# This need to be pre-generated depending on state features
with open("./agent_code/agent_SUN/model/state_to_index.pk", "rb") as file:
    state_to_index = pickle.load(file)

def nudge_direction(nudge):
    flag = np.array([nudge[1]==-1,nudge[0]==1,nudge[1]==1,nudge[0]==-1,nudge[0]==nudge[1]==0]) # if no more target, WAIt
    action_idx = np.arange(5)[flag].astype(int)
    return action_idx


# ONLY NEED TO RUN ONCE
# from itertools import product
# def initialize_dict():
#     state_to_index = dict()
#     index_to_state = dict()
#     counter = 0
#     for i,v in enumerate(vicinity_set):
#         neighbours = tuple(product([-1, 0, 1, 2, 3], repeat=len(v.neighbours)))
#         nudge = tuple(product([-1, 0, 1], repeat=2))
#         # state = (vicinity_type,(neighbours),(nudge),b_indicator) with all possible permutations
#         for n1,n2 in tuple(product(neighbours,nudge)):
#             state = (i,n1,n2,True)
#             state_to_index[state] = counter
#             index_to_state[counter] = state
#             state = (i,n1,n2, False)
#             state_to_index[state] = counter+1
#             index_to_state[counter+1] = state
#             counter +=2
#     with open("model/state_to_index.pk", "wb") as file:
#         pickle.dump(state_to_index, file)
#     with open("model/index_to_state.pk", "wb") as file:
#         pickle.dump(state_to_index, file)
#     # len(state_to_index) #2550 -> #22950
#     # len(index_to_state) #2550 *9


### OLD <version 1> state = (vicinity_type,(neighbours)), NO BOMB INDICATOR
# def initialize_Q():
#     Q = np.zeros([len(state_to_index),len(ACTIONS)])-np.Inf
#     for k,v in state_to_index.items():
#         actions = np.array(vicinity_set[k[0]].actions) #all legal moves defined by vicinity type
#         # customized by actual surrounding situation, only truly allowed if free_tile(0) OR coin(2)
#         mask = np.logical_or(np.array(k[1]) == 0,np.array(k[1]) == 2)
#         mask = np.append(mask, [True,True]) # for WAIT and BOMB
#         valid_actions = actions[mask]
#         Q[v][valid_actions] = 0
#     # np.sum(Q == -np.Inf) #7650 --> #3380
#     return Q

### OLD <version 2> state = (vicinity_type,(neighbours),b_indicator), NO directional NUDGE
# def initialize_Q():
#     Q = np.zeros([len(state_to_index),len(ACTIONS)])-np.Inf
#     for k,v in state_to_index.items():
#         # k = (vicinity_type,(neighbours),b_indicator)
#         actions = np.array(vicinity_set[k[0]].actions) #all legal moves defined by vicinity type
#         # customized by actual surrounding situation, only truly allowed if free_tile(0) OR coin(2)
#         mask = np.logical_or(np.array(k[1]) == 0,np.array(k[1]) == 2)
#         mask = np.append(mask, [True]+[k[2]]) # True for WAIT; k[2] is BOMB_indicator
#         valid_actions = actions[mask]
#         Q[v][valid_actions] = 0
#     # np.sum(Q == -np.Inf) #15300 --> #8035 = 3380*2 + 2550/2
#     return Q


# NEW <version 3> state = (vicinity_type,(neighbours),(nudge),b_indicator)
def initialize_Q():
    Q = np.zeros([len(state_to_index),len(ACTIONS)])-np.Inf
    for k,v in state_to_index.items():
        # k = (vicinity_type,(neighbours),(nudge),b_indicator)
        actions = np.array(vicinity_set[k[0]].actions) #all legal moves defined by vicinity type
        # customized by actual surrounding situation, only truly allowed if free_tile(0) OR coin(2)
        mask = np.logical_or(np.array(k[1]) == 0,np.array(k[1]) == 2)
        mask = np.append(mask, [True]+[k[-1]]) # True for WAIT; k[-1] is BOMB_indicator
        valid_actions = actions[mask]
        Q[v][valid_actions] = 0
    # np.sum(Q == -np.Inf) #137700 --> #72315 = 8035 *9
    return Q



def setup(self):
    """
    Setup your code. This is called once when loading each agent. Make sure that you prepare everything such that act(...) can be called.
    When in training mode, the separate `setup_training` in train.py is called after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.
    In this example, our model is a set of probabilities over actions that are is independent of the game state.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.vicinity_type = vicinity_type
    self.vicinity_set = vicinity_set
    self.state_to_index = state_to_index
    self.eps = eps
    self.tracker = deque(maxlen=TRACKER_HISTORY)

    if self.train or not os.path.isfile(Q_save):
        self.logger.info("Initiate Q-table for training.")
        self.Q = initialize_Q()
    else:
        self.logger.info("Loading saved Q-table from previous training.")
        with open(Q_save, "rb") as file:
            self.Q = pickle.load(file)

    ####### for record keeping and experiment analysis REMOVE BEFORE COMPETITION TODO
    self.round = 0
    self.step = 0
    self.score = 0
    self.record = []
    #######


def act(self, game_state: dict) -> str:
    """ Your agent should parse the input, think, and take a decision. When not in training mode, the maximum execution time for this method is 0.5s.
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string. """

    ###### for record keeping and experiment analysis REMOVE BEFORE TRAINING TODO python main.py play --agents agent_SUN --n-round 11 --no-gui

    if game_state['round'] == self.round+1:
        self.record.append([self.round, self.step, self.score])
        self.round = game_state['round']
    else:
        self.step = game_state['step']
        self.score = game_state['self'][1]

    if len(self.record) == 11:
        np.savetxt(f'{path}score_card_{self.round-1}_rounds{version}_eps{self.eps}.csv', self.record, delimiter=',')

    ######


    self.tracker.append(game_state['self'][-1]) # track 20 past locations in history
    loop = self.tracker.count(self.tracker[-1]) >= 3

    features, allowed_actions = state_to_features(self, game_state)
    stateIndex = features_to_stateIndex(self, features)

    actionIndex = np.argmax(self.Q[stateIndex]) # best policy according to Q-table

    if self.Q[stateIndex,actionIndex] == 0 or np.random.rand() < self.eps: # state-action never seen before, explore, take a random selection
        explore = True
    else:
        explore = False

    # in training we want our agent to explore all possible states; in actual games, agent should follow Q-table and if never seen then explore
    if self.train:
        explore = np.random.rand() < self.eps_train # initially always explores during training then switch to follow Q with some epsilon

    # when gaming (i.e. not in training mode), just take the best policy from Q-table --> this sometimes got stuck: not moving or in a loop
    # if not self.train:
    #     return ACTIONS[actionIndex]

    # NEW - to avoid laziness in movement
    # take any other allowed actions according to Q-table if location not changed in the last 3 moves
    # lazy = (len(self.tracker)>3) and (actionIndex == 4) and (self.tracker[-1] == self.tracker[-2] == self.tracker[-3] == self.tracker[-4])
    # if current proposal is 'WAIT' and agent's location has not changed in the last 3 moves

    # If agent has been in the same location three times recently, it's a loop

    # if not self.train:
    #     explore = np.random.rand() < self.eps  # allow some random exploration
    # else: # when in training mode,
    #     explore = np.random.rand() < self.eps_train

    if loop:
        # actionIndex = np.argsort(self.Q[stateIndex])[-2] # take the second best option
# TODO
#         allowed_actions = np.array(allowed_actions)[np.array(allowed_actions) < 4]  # <4 NO BOMB NO WAIT for coins only
        allowed_actions = np.array(allowed_actions)[np.array(allowed_actions) != 4]  # NO WAIT, allow BOMB for crates
        # actionIndex = random.choice(allowed_actions) # take a random action within the allowed scope
        preferred = [i for i in nudge_direction(nudge=features[2]) if i in allowed_actions]
        if explore:
            actionIndex = random.choice(allowed_actions)
        else:
            actionIndex = random.choice(preferred) if len(preferred)>0 else random.choice(allowed_actions) # choose a nudge direction if possible or a random direction if not possible

    self.logger.info(f"\n >>> Step:{game_state['step']}")
    self.logger.info(f"\n{self.global_map}")
    self.logger.info(
        f"State: {[features_to_stateIndex(self, features)], features, [ACTIONS[i] for i in allowed_actions]}")
    self.logger.info(f"Action: {ACTIONS[actionIndex]}, Q-values: {self.Q[stateIndex]}")

        # if np.random.rand() < self.eps_train:
        #     # actionIndex = np.random.randint(len(ACTIONS))
        #     # rather than taking random action, we choose from only valid actions to make training more efficient
        #     actionIndex = random.choice(allowed_actions) # this sometime explore possible movement according

        # print('\n Step:',game_state['step'])
        # print(self.global_map)
        # if self.minefield.max()!= -SAFE+1:
        #     print(self.minefield)
        # print('State:', [features_to_stateIndex(self,features)], features, [ACTIONS[i] for i in allowed_actions], 'Exploring:',np.random.rand() < self.eps)
        # print('Action: ', ACTIONS[actionIndex])

        # Training without BOMB --> WAIT; otherwise each round finishes too quickly due to bombing itself
        # if self.pacifier:
        #     if ACTIONS[actionIndex] == 'BOMB':
        #         return 'WAIT'

    return ACTIONS[actionIndex]


def project_single_blast(self,x,y,t):
    """ create a single layer of blast map for an active bomb;
    returns a np.array with explosion range marked by remaining time to explosion +t, safe tiles are marked with SAFE (a high number 99)
    """
    _layer = np.ones([s.ROWS,s.COLS])*SAFE #initializing safe_tiles with a high number
    vicinity_index = (self.vicinity_type[y,x]).astype(int)
    xmin,xmax,ymin,ymax = self.vicinity_set[vicinity_index].explosion_range
    _layer[y+ymin:y+ymax+1,x] = t
    _layer[y,x+xmin:x+xmax+1] = t
    return _layer

def project_all_blasts(self, bombs,n):
    """project a combined blast map in n-timesteps with all bombs"""
    _layer = np.ones([s.ROWS,s.COLS])*SAFE #initializing safe_tiles with a high number
    bomb_layers = [_layer]
    active_bombs = [b for b in bombs if b[1] > n]
    for ((x, y), t) in active_bombs:
        _layer = project_single_blast(self,x, y, t-n) # adjust for projected blast at forward n-timesteps
        bomb_layers.append(_layer)
    bombs_map = np.stack(bomb_layers,axis=0).min(axis=0) # immediate explosion is more dangerous than future-dated
    return bombs_map


def features_to_stateIndex(self, features):
    '''map features into int index for accessing Q-table, based on pre-generated dict self.state_to_index'''
    if features is None:
        return None
    stateIndex = self.state_to_index[features]
    return stateIndex


def create_global_map(self,game_state):
    '''Create a world map (np.array) based on game_state'''
    if game_state is None:
        self.global_map = None

    # ’field’: np.array(width, height) describing the tiles of the game board. Its entries are 1 for crates, −1 for stone walls and 0 for free tiles.
    field = game_state['field'].T.copy()

    crates = [(x, y) for x in range(1, s.COLS-1) for y in range(1, s.ROWS-1) if (field[x, y] == 1)]

    # list of coins cordinates
    coins = game_state['coins']

    # enermy agent's location
    others = game_state['others']
    enemies = [agent[-1] for agent in others]

    self.treasure_map = field

    # The value in each tile represents the current state {-1: wall, 0: free_tile, 1: crate, 2: coin, 3: enemy_agent} - layer1
    for coin in coins:
        self.treasure_map[coin[1], coin[0]] = 2
    for enemy in enemies:
        self.treasure_map[enemy[1], enemy[0]] = 3

    # 'explosion_map': np.array(width, height) stating for each tile how many more steps an explosion will be present. Where there is no explosion, the value is 0.
    explosion_map = game_state['explosion_map']
    # modify the explosion map so that the safe_tiles are marked with a high number instead of zero
    explosion_map[explosion_map == 0] = SAFE

    # 'bombs': list of tuples [((x, y), t),...] of coordinates and countdowns for all active bombs.
    bombs = game_state['bombs']
    bombs_map = project_all_blasts(self, bombs, n=0)  # current

    for bomb in bombs:
        self.treasure_map[bomb[0][1], bomb[0][0]] = -1 # update bomb as an obstacle {-1} in treasure map so do not step onto it

    # overlay ticking bombs on explosion map to create minefield

    # OLD
    # minefield is a field-like dict with timer indicating time-steps to safety clearance, e.g.
    # {-1: lingering smoke from previous explosion (1-timestep to clearance, corresponding to explosion_map), this is HARMLESS
    #  -2: bomb exploding (2-timestep to clerance, corresponding ((x,y) t = 0), -t-2),
    #  ...
    #  -5: bomb just dropped in the last step, ((x,y) t = 3), -t-2}
    # negative values to differentiate from other objects {-1: wall, 0: free_tile, 1: crate, 2: coin, 3: enemy}
    # global_map contains ticking bombs t-2 = {-5,-4,-3,-2}, explosion occurs at bomb.t = 0 and explosion.t=2,
    # both corresponds to {-2} on global_map
    # minefield = np.maximum(explosion_map * -1, bombs_map * -1 - 2)
    # self.global_map = ((minefield == -SAFE) * treasure_map + (minefield != -SAFE) * minefield)


    # NEW
    # explosion.t = 1 is harmless smoke, it does not kill agent stepping into it -> need to convert minefield {-1} --> global {0}, equivalent to free-tile or whatever masked by the smoke
    # minefield is a field-like dict with timer indicating time-steps to safety clearance, e.g.
        # {0: lingering smoke from previous explosion (indicator of safety - HARMLESS), corresponding blast_coords in explosion_map ((x,y), explosion.t=1)); bomb no longer active
        #  -1: bomb exploding (bomb.t = 0, explosion.t = 2, indicating 1-timestep to safety, corresponding blast_coords in explosion_map ((x,y), explosion.t=2)),
        #      this is where the most events and rewards are handed out: e.BOMB_EXPLODED, e.CRAFT_DESTROYED, e.COIN_FOUND, e.AGENT_KILLED etc.
        #  ...
        #  -4: bomb just dropped in the last step, corresponding to active bombs ((x,y) bomb.t = 3)
        #  -SAFE+1 or -SAFE-1: regions not affected by explosion /bombs }
    # bomb.t * -1 - 1  ==  explosion.t * -1 + 1
    self.minefield = np.maximum(explosion_map * -1 + 1 , bombs_map * -1 - 1)

    # combine minefield with field_with_, minefield overwrites other objects
    self.global_map = ((self.minefield < -4) * self.treasure_map + (self.minefield >= -4) * self.minefield)

    # my own agent's coordinates
    x, y = game_state['self'][-1]

    # update my position in global map
    self.global_map[y,x] = 7

    # targets list
    self.targets = coins + crates + enemies

    return self



roi_dict = {(0,-1):[-2,0,-2,3],
     (1,0):[-2,3,1,3],
     (0,1):[0,3,-2,3],
     (-1,0):[-2,3,-2,0]}

# NEW -- trying to speed up take a rough measure of free-tile density rather than precise region
def infer_local_features(self, agent):
    """agent: A tuple (name, score, b_indicator,(x, y)) describing an agent: game_state['self] or game_state['others'][i]
    return: a tuple (feature, allowed actions) where feature -> tuple(vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """
    x, y = agent[-1]
    b_indicator = agent[2]
    vicinity_index = (self.vicinity_type[y, x]).astype(int)
    vicinity = self.vicinity_set[vicinity_index] #Vicinity = namedtuple('Vicinity', ['neighbours', 'explosion_range', 'actions'])
    neighbours = [(x + ij[0], y + ij[1]) for ij in vicinity.neighbours] # get global coordinates of neighbours
    local_neighbourhood = np.array([self.treasure_map[y_, x_] for (x_, y_) in neighbours]) # get identities of neighbours from treasure map
    # boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local environment
    m_indicator = np.logical_or(local_neighbourhood==0, local_neighbourhood==2)  # np.array[True,False] corresponding to vicinity.actions (already restricted)

    # initializing nudge here to ensure it exists
    nudge = np.array([0, 0])

    # start = time.time()

    # OLD ADDED 2021.03.26
    # rather than finding the precise escape routes, scan the neighbouring region through neighbour; if free-paths exists then bomb, otherwise don't.

    # roi = self.treasure_map[max(0, y-2):min(s.ROWS-1, y+3), max(0, x-2):min(s.COLS-1, x+3)]  # region of interest
    # safe = (np.sum(roi == 0) + np.sum(roi == 2)) > roi.size * 0.25  # (np.sum(roi == -1) + np.sum(roi == 1)+ np.sum(roi == 3)+1) # if free tile is more than non-free
    #
    # roi_UP = self.treasure_map[max(0, y-2):min(s.ROWS-1, y), max(0, x-2):min(s.COLS-1, x+3)]
    # roi_RIGHT = self.treasure_map[max(0, y-2):min(s.ROWS-1, y+3), max(0, x+1):min(s.COLS-1, x+3)]
    # roi_DOWN = self.treasure_map[max(0, y):min(s.ROWS-1, y+3), max(0, x-2):min(s.COLS-1, x+3)]
    # roi_LEFT = self.treasure_map[max(0, y-2):min(s.ROWS-1, y+3), max(0, x-2):min(s.COLS-1, x)]
    #
    # dof_UP = (np.sum(roi_UP == 0) + np.sum(roi_UP == 2))
    # dof_RIGHT = (np.sum(roi_RIGHT == 0) + np.sum(roi_RIGHT == 2))
    # dof_DOWN = (np.sum(roi_DOWN == 0) + np.sum(roi_DOWN == 2))
    # dof_LEFT = (np.sum(roi_LEFT == 0) + np.sum(roi_LEFT == 2))
    #
    # if not safe:
    #     b_indicator = False
    # if b_indicator == False: # meaning recently dropped a bomb, nudge indicator should prioritize escape route
    #     # select the 'safest' option, i.e. direction with more free-tiles, if more than one, then random select
    #     nudge = np.sign([dof_RIGHT - dof_LEFT, -(dof_UP - dof_DOWN)])



    # NEWLY ADDED 2021.03.27 - make use of vicinity.neighbours (tuple of relative coordinates (i,j) for only non-wall neighbours)
    safe = [0]
    direction = [(0,0)]

    for sn_ij in np.array(vicinity.neighbours)[m_indicator]:  # list of global coords of safe neighbours only
        j_0, j_1, i_0, i_1 = roi_dict[tuple(sn_ij)]
        roi_ = self.treasure_map[max(0, int(y + j_0)):min(s.ROWS - 1, int(y + j_1)), max(0, int(x + i_0)):min(s.COLS - 1, int(x + i_1))]
        dof_ = (np.sum(roi_ == 0) + np.sum(roi_ == 2)) / roi_.size
        safe.append(dof_)
        direction.append(sn_ij)

    # print(vicinity.neighbours, m_indicator, safe, direction)

    if max(safe) < 0.33: # if less than 2 free tile on region of 6 then not safe
        b_indicator = False

    if b_indicator == False: # meaning recently dropped a bomb, nudge indicator should prioritize escape route
        # select the 'safest' option, i.e. direction with more free-tiles, if more than one, then random select
        nudge = np.array(direction[np.argmax(safe)])
        # nudge = np.array([0,0])
    # nudging towards closest target, if targets is not a empty list
    elif len(self.targets) > 0:
        d = np.array(self.targets) - np.array([x, y])
        min_idx = d.sum(axis=1).argmin()  # index for nearest target based on abs(dx)+abs(dy)
        self.nearest_target = d[min_idx].sum()  # record this for auxiliary award
        nudge = np.sign(d[min_idx])

    # if b_indicator == False: # meaning recently dropped a bomb, nudge indicator should prioritize escape route
    #     # select the 'safest' option, i.e. direction with max num of escape routes, if more than one, then random select
    #     safe_idx = random.choice(np.where(escape_routes == max(escape_routes)))
    #     nudge = np.sign(vicinity.neighbours[safe_idx])  # this is the relative direction

    # print('...timing:',time.time()-start,'safe =',safe,'b_indicator=',b_indicator)

    features = (vicinity_index, tuple(local_neighbourhood), tuple(nudge), b_indicator) #bomb-indicator added as an additional feature; # nudge added as an additiional feature

    allowed_actions = np.array(vicinity.actions)[np.append(m_indicator, [True] + [b_indicator])]
        # self.vicinity_set[vicinity_index].actions -> list [0,1,2,3,4,5]
        # m_indicator -> np.array[True,False,True,True]
        # vicinity.actions already pre-screened and only contains valid actions
        # allowed_actions is subset of a numpy array of [0,1,2,3,4,5] of variable length 4-6

    return features, allowed_actions


# OLD -- trying to speed up // added escape route to overwrite b_indicator and nudge_indicator
# '''dictionary returns a region of 6 tiles relative to the reference point, key = (i,j) relative direction, value = [x_min,x_max,y_min,y_max]'''
# region_of_interest = dict()
# region_of_interest[(0,-1)] = [-1,1,-2,-1] # top
# region_of_interest[(0,1)] = [-1,1,1,2]   # bottom
# region_of_interest[(-1,0)] = [-2,-1,-1,1] # left
# region_of_interest[(1,0)] = [1,2,-1,1] # right

# def infer_local_features(self, agent):
#     """agent: A tuple (name, score, b_indicator,(x, y)) describing an agent: game_state['self] or game_state['others'][i]
#     return: a tuple (feature, allowed actions) where feature -> tuple(vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """
#     x, y = agent[-1]
#     b_indicator = agent[2]
#     vicinity_index = (self.vicinity_type[y, x]).astype(int)
#     vicinity = self.vicinity_set[vicinity_index] #Vicinity = namedtuple('Vicinity', ['neighbours', 'explosion_range', 'actions'])
#     neighbours = [(x + ij[0], y + ij[1]) for ij in vicinity.neighbours] # get global coordinates of neighbours
#
#     local_neighbourhood = np.array([self.treasure_map[y_, x_] for (x_, y_) in neighbours]) # get identities of neighbours from treasure map
#
#     # boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local environment
#     m_indicator = np.logical_or(local_neighbourhood==0, local_neighbourhood==2)  # np.array[True,False] corresponding to vicinity.actions (already restricted)
#
#     # initializing nudge here to ensure it exists
#     nudge = np.array([0, 0])
#
#     # NEWLY ADDED 2021.03.26
#     start = time.time()
#     # rather than finding the precise escape routes, scan the neighbouring region through neighbour; if free-paths exists then bomb, otherwise don't.
#     # STILL TOO SLOW...
#     if vicinity_index in [1,2]:
#         n_free = 2
#     else:
#         n_free = 3
#     # for type 1 and 2 vinicity, min 2 steps to safety; other types min 3 steps
#
#     escape_routes = [] # count free-tile density in regions through all neighbours
#
#     for (sn_ij, m) in zip(np.array(vicinity.neighbours), m_indicator):  # global, local coordinates and safety-indicator of all neighbours
#         if not m:
#             # m_indicator =FALSE means there is an obstacle (crates, enemy, bomb) currently occupying the neighbour, therefore note safe / impossible to escape via this neighbour
#             escape_routes.append(0)
#         else:
#             x_0,x_1,y_0,y_1 = [x,x,y,y] + region_of_interest[sn_ij]
#             roi = self.treasure_map[min(0,y_0):max(16,y_1+1), min(0,x_0):max(16,x_1+1)] # region of interest
#             safe = (np.sum(roi == 0) + np.sum(roi == 2))
#             if safe >= n_free:
#                 escape_routes.append(safe)
#             else:
#                 escape_routes.append(0)
#
#     # escape_routes = [0,0,0,0]
#     if np.array(escape_routes).sum() == 0: # no escape route available # disallow bombing
#         b_indicator = False
#         # nudging towards closest target, if targets is not a empty list
#         if len(self.targets) > 0:
#             d = np.array(self.targets) - np.array([x, y])
#             min_idx = d.sum(axis=1).argmin()  # index for nearest target based on abs(dx)+abs(dy)
#             self.nearest_target = d[min_idx].sum()  # record this for auxiliary award
#             nudge = np.sign(d[min_idx])
#
#     if b_indicator == False: # meaning recently dropped a bomb, nudge indicator should prioritize escape route
#         # select the 'safest' option, i.e. direction with max num of escape routes, if more than one, then random select
#         safe_idx = random.choice(np.where(escape_routes == max(escape_routes)))
#         nudge = np.sign(vicinity.neighbours[safe_idx])  # this is the relative direction
#
#     print('...timing:',time.time()-start)
#
#     features = (vicinity_index, tuple(local_neighbourhood), tuple(nudge), b_indicator) #bomb-indicator added as an additional feature; # nudge added as an additiional feature
#
#     allowed_actions = np.array(vicinity.actions)[np.append(m_indicator, [True] + [b_indicator])]
#         # self.vicinity_set[vicinity_index].actions -> list [0,1,2,3,4,5]
#         # m_indicator -> np.array[True,False,True,True]
#         # vicinity.actions already pre-screened and only contains valid actions
#         # allowed_actions is subset of a numpy array of [0,1,2,3,4,5] of variable length 4-6
#
#     print(f'...calling infer_local_features()...escape={escape_routes}, features={features}, actions={[ACTIONS[i] for i in allowed_actions]}')
#     self.logger.info(f"...calling infer_local_features()...escape={escape_routes}")
#     return features, allowed_actions
#



# # OLD -- added escape route to overwrite b_indicator and nudge_indicator -- too slow
# def infer_local_features(self, agent):
#     """agent: A tuple (name, score, b_indicator,(x, y)) describing an agent: game_state['self] or game_state['others'][i]
#     return: a tuple (feature, allowed actions) where feature -> tuple(vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """
#     x, y = agent[-1]
#     b_indicator = agent[2]
#     vicinity_index = (self.vicinity_type[y, x]).astype(int)
#     vicinity = self.vicinity_set[vicinity_index] #Vicinity = namedtuple('Vicinity', ['neighbours', 'explosion_range', 'actions'])
#     neighbours = [(x + ij[0], y + ij[1]) for ij in vicinity.neighbours] # get global coordinates of neighbours
#
#     # global_map contains ticking bombs t-2 = {-5,-4,-3,-2}, explosion occurs at bomb.t = 0 and explosion.t=2, both corresponds to {-2} on global_map
#     # explosion.t = 1 is harmless smoke, it does not kill agent when stepping into it --> need to convert minefield {-1} --> global {0}, equivalent to free-tile or whatever masked by the smoke
#
#     # NO LONGER in USE previous logic: only convert {-2,-1} -> {-1}: future bombs does not represent immediate risk for next step (i.e. can always step back in the next state)
#     # local_neighbourhood = local_neighbourhood*(local_neighbourhood>-2) - 1*(local_neighbourhood==-2)
#     # OLD
#     # local_neighbourhood = np.array([self.global_map[j, i] for (i, j) in neighbours])
#     # local_neighbourhood = local_neighbourhood*(local_neighbourhood>-2) - 1*(local_neighbourhood==-2)
#
#     # NEW use treasure_map as a guidance
#     local_neighbourhood = np.array([self.treasure_map[y_, x_] for (x_, y_) in neighbours]) # get identities of neighbours from treasure map
#
#     # boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local environment
#     m_indicator = np.logical_or(local_neighbourhood==0, local_neighbourhood==2)  # np.array[True,False] corresponding to vicinity.actions (already restricted)
#
#     # initializing nudge here to ensure it exists
#     nudge = np.array([0, 0])
#
#     # NEWLY ADDED 2021.03.26
#     # Scan through neighbours for escape routes; if exists then bomb, otherwise don't.
#     escape_routes = [] # count number of safe routes in all neighbours
#     # escape_routes: list of same length as vicinity.neighbour and m_indicator
#     for (sn_xy, sn_ij, m) in zip(np.array(neighbours), np.array(vicinity.neighbours), m_indicator):  # global, local coordinates and safety-indicator of all neighbours
#         if not m:
#             # m_indicator =FALSE means there is an obstacle (crates, enemy, bomb) currently occupying the neighbour, therefore note safe / impossible to escape via this neighbour
#             escape_routes.append(0)
#         else:
#             # there are generally 3 routes through the immediate neighbour
#             route_1 = [sn_xy + sn_ij * i for i in range(1, 4)] # straight: need extra 3 steps in addition to neighbour tile
#             route_2 = [sn_xy + sn_ij, sn_xy + sn_ij + 1 * np.array(sn_ij == 0)] # through neighbour then turn 90 or -90 degree
#             route_3 = [sn_xy + sn_ij, sn_xy + sn_ij + (-1) * np.array(sn_ij == 0)]
#             route_4 = [sn_xy + 1 * np.array(sn_ij == 0)]
#             route_5 = [sn_xy - 1 * np.array(sn_ij == 0)]
#
#             safe_route_counter = 0
#             # after getting all the global coordinates for possible routes, we check for the object idenity from treasure map, only TRUE for freedom to move on each route
#             for route_coords in [route_1, route_2, route_3, route_4,route_5,]:
#                 route_obj = np.array([self.treasure_map[valid_boundary(y_), valid_boundary(x_)] for (x_, y_) in route_coords])
#                 safe = np.all(np.logical_or(route_obj == 0, route_obj == 2))
#                 safe_route_counter += safe
#             escape_routes.append(safe_route_counter)
#
#     # escape route = [0,0,0,0]
#     if np.array(escape_routes).sum() == 0: # no escape route available # disallow bombing
#         b_indicator = False
#         # nudging towards closest target, if targets is not a empty list
#         # OLD COMMENT add a nudge to features: nudge = [i,j] where i,j = {-1,0,+1} indicating where the nearest target is relative to agent
#         if len(self.targets) > 0:
#             d = np.array(self.targets) - np.array([x, y])
#             min_idx = d.sum(axis=1).argmin()  # index for nearest target based on abs(dx)+abs(dy)
#             self.nearest_target = d[min_idx].sum()  # record this for auxiliary award
#             nudge = np.sign(d[min_idx])
#
#     if b_indicator == False: # meaning recently dropped a bomb, nudge indicator should prioritize escape route
#         # select the 'safest' option, i.e. direction with max num of escape routes, if more than one, then random select
#         safe_idx = random.choice(np.where(escape_routes == max(escape_routes)))
#         nudge = np.sign(vicinity.neighbours[safe_idx])  # this is the relative direction
#
#     # else: # escape route is available and bomb is possible, then nudging should not matter.
#     #     pass
#
#     features = (vicinity_index, tuple(local_neighbourhood), tuple(nudge), b_indicator) #bomb-indicator added as an additional feature; # nudge added as an additiional feature
#
#     allowed_actions = np.array(vicinity.actions)[np.append(m_indicator, [True] + [b_indicator])]
#         # self.vicinity_set[vicinity_index].actions -> list [0,1,2,3,4,5]
#         # m_indicator -> np.array[True,False,True,True]
#         # vicinity.actions already pre-screened and only contains valid actions
#         # allowed_actions is subset of a numpy array of [0,1,2,3,4,5] of variable length 4-6
#
#     print(f'...calling infer_local_features()...escape={escape_routes}, features={features}, actions={[ACTIONS[i] for i in allowed_actions]}')
#     self.logger.info(f"...calling infer_local_features()...escape={escape_routes}")
#     return features, allowed_actions

### OLD - fast but bomb without checking escape route

# def infer_local_features(self, agent):
#     """agent: A tuple (name, score, b_indicator,(x, y)) describing an agent: game_state['self] or game_state['others'][i]
#     return: a tuple (feature, allowed actions) where feature -> tuple(vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """
#     x, y = agent[-1]
#     b_indicator = agent[2]
#     vicinity_index = (self.vicinity_type[y, x]).astype(int)
#     neighbours = [(x + ij[0], y + ij[1]) for ij in self.vicinity_set[vicinity_index].neighbours]
#
#     # global_map contains ticking bombs t-2 = {-5,-4,-3,-2}, explosion occurs at bomb.t = 0 and explosion.t=2, both corresponds to {-2} on global_map
#     # explosion.t = 1 is harmless smoke, it does not kill agent when stepping into it --> need to convert minefield {-1} --> global {0}, equivalent to free-tile or whatever masked by the smoke
#
#     # previous logic NO LONGER in USE: only convert {-2,-1} -> {-1}: future bombs does not represent immediate risk for next step (i.e. can always step back in the next state)
#     # local_neighbourhood = local_neighbourhood*(local_neighbourhood>-2) - 1*(local_neighbourhood==-2)
#     # OLD
#     # local_neighbourhood = np.array([self.global_map[j, i] for (i, j) in neighbours])
#     # local_neighbourhood = local_neighbourhood*(local_neighbourhood>-2) - 1*(local_neighbourhood==-2)
#
#     # NEW use treasure_map as a guidance
#     local_neighbourhood = np.array([self.treasure_map[j, i] for (i, j) in neighbours])
#
#     # add a nudge to features: nudge = [i,j] where i,j = {-1,0,+1} indicating where the nearest target is relative to agent
#     if len(self.targets) > 0:
#         d = np.array(self.targets) - np.array([x, y])
#         min_idx = d.sum(axis=1).argmin()  # index for nearest target based on abs(dx)+abs(dy)
#         self.nearest_target = d[min_idx].sum() # record this for auxiliary award
#         nudge = np.sign(d[min_idx])
#     else:
#         nudge = np.array([0,0])
#     features = (vicinity_index, tuple(local_neighbourhood), tuple(nudge), b_indicator) #bomb-indicator added as an additional feature; # nudge added as an additiional feature
#
#     m_indicator = np.logical_or(local_neighbourhood==0, local_neighbourhood==2) #boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local enviroment
#
#     allowed_actions = np.array(self.vicinity_set[vicinity_index].actions)[np.append(m_indicator, [True] + [b_indicator])]
#         # self.vicinity_set[vicinity_index].actions -> list [0,1,2,3,4,5]
#         # m_indicator -> np.array[True,False,True,True]
#         # allowed_actions is subset of a numpy array of [0,1,2,3,4,5] of variable length 4-6
#
#     return features, allowed_actions
#





# Not in Use -> Decided to use separate training agents instead of observing from my agent's perspective (no easy way to get e.EVENTS of enemies)
# def get_enemy_features(self, game_state: dict) -> list:
#     """
#     :param game_state:  A dictionary describing the current game board.
#     :return: a list of tuples each representing an agent (feature, allowed actions) where feature = (vicinity_type,(neighbours),(b_indicator)), allowed actions ->list
#     the [0] is own agent; the [1:] are others
#     """
#     if game_state is None or game_state['others'] == []:
#         return None
#
#     # synthesize a global map from game states, i.e. self.global_map is created
#     create_global_map(self,game_state)
#
#     # get local features for each agent
#
#     enemy_features = [infer_local_features(self, agent) for agent in game_state['others']]
#     # print(game_state['others'], game_state['others'][0][-1])
#
#     # in case of encounter our agent, replace 7 with 3
#     enemy_features_ = []
#     for enemy in enemy_features:
#         if 7 in enemy[0][1]:
#             t = enemy[0]
#             n = list(t[1])
#             n[n.index(7)] = 3
#             t = t[0], tuple(n), t[2]
#             enemy = (t,enemy[1])
#         enemy_features_.append(enemy)
#     return enemy_features_



def state_to_features(self, game_state: dict) -> tuple:
    """ Converts the game state to the input of your model, i.e. a feature vector.
    :param game_state:  A dictionary describing the current game board.
    :return: a tuple (feature, allowed actions) where feature = (vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """

    # start = time.time()
    if game_state is None:
        return None,None  # This needs to conform to return below

    create_global_map(self,game_state)
    my_agent = game_state['self']

    # if time.time() - start > 0.5:
    #     print(f">>> ...infer time: {time.time() - start}")

    features, allowed_actions = infer_local_features(self, my_agent)

    return features, allowed_actions


# Not in Use -> Decided to split into different functions and shorten this function
# def state_to_features(self, game_state: dict) -> tuple:
#     """
#     *This is not a required function, but an idea to structure your code.*
#     Converts the game state to the input of your model, i.e. a feature vector.
#     You can find out about the state of the game environment via game_state, which is a dictionary.
#     Consult 'get_state_for_agent' in environment.py to see what it contains.
#     :param game_state:  A dictionary describing the current game board.
#     :return: a tuple (feature, allowed actions) where feature = (vicinity_type,(neighbours),(b_indicator)), allowed actions ->list
#     """
#
#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None,None
#         # This needs to conform to return below
#
# # Create a world map based on game_state
#
#     # ’field’: np.array(width, height) describing the tiles of the game board. Its entries are 1 for crates, −1 for stone walls and 0 for free tiles.
#     field = game_state['field']
#
#     # list of coins cordinates
#     coins = game_state['coins']
#
#     # enermy agent's location
#     others = game_state['others']
#     enemies = [agent[-1] for agent in others ]
#
#     treasure_map = field.copy()  # 17*17 = 289, 176 free tiles
#     # The value in each tile represents the current state {-1: wall, 0: free_tile, 1: crate, 2: coin, 3: enemy_agent} - layer1
#     for coin in coins:
#         treasure_map[coin[1],coin[0]] = 2
#     for enemy in enemies:
#         treasure_map[enemy[1],enemy[0]] = 3
#
#     # 'explosion_map': np.array(width, height) stating for each tile how many more steps an explosion will be present. Where there is no explosion, the value is 0.
#     explosion_map = game_state['explosion_map']
#     # modify the explosion map so that the safe_tiles are marked with a high number instead of zero
#     explosion_map[explosion_map == 0] = SAFE
#
#     # 'bombs': list of tuples [((x, y), t),...] of coordinates and countdowns for all active bombs.
#     bombs = game_state['bombs']
#     bombs_map = project_all_blasts(self,bombs,n=0) # current
#
#     # overlay ticking bombs on explosion map to create minefield
#     # {-1: lingering explosion (1-timestep to clearance, corresponding to explosion_map),
#     #  -2: currently exploding (2-timestep to clerance, corresponding to bomb timer = 0), ...}
#     # negative values to differentiate from other objects {-1: wall, 0: free_tile, 1: crate, 2: coin, 3: enemy}
#     minefield = np.maximum(explosion_map * -1, bombs_map * -1 - 2)
#
#     # combine minefield with field_with_, minefield overwrites other objects
#     self.global_map = ((minefield == -SAFE) * treasure_map + (minefield != -SAFE) * minefield)
#
# # Create a local map based on my agent's own coordinate
#     # my own agent's coordinates
#     x, y = game_state['self'][-1]
#     # update my position in global map
#     self.global_map[y,x] = 7
#
#
#     # bomb indicator
#     b_indicator = game_state['self'][2]
#
#     # Agent looking around to check out the immediate neighbourhood; immediate neighbourhood is defined as tiles at [(x,y-1), (x+1,y), (x,y+1), (x-1,y)] relative to agent's coordinates (x,y),
#     # which also corresponding to the next possible agent's position with actions [UP, RIGHT, DOWN, LEFT]
#     # The value in each tile represents the current state {-1: wall, 0: free_tile, 1: crate, 2: coin, 3: enemy_agent} - layer1
#     # Separate layer for minefield {-1: explosion persisting or bomb about to explode in the next time step} - layer2
#     # priority order: 1) avoid minefield, 2) drop bomb when crate and/or enemy in neighbourhood, 3) collect coin if any, 4) if escape route available, drop bomb.
#
#     vicinity_index = (self.vicinity_type[y, x]).astype(int)
#     neighbours = [(x + ij[0], y + ij[1]) for ij in self.vicinity_set[vicinity_index].neighbours]
#     local_neighbourhood = np.array([self.global_map[j, i] for (i, j) in neighbours])
#
#     # global_map contains ticking bombs [-t-2...-1], need to simplify: only convert [-2,-1] to [-1]; future bombs does not represent immediate risk for next step (i.e. can always step back in the next state)
#     local_neighbourhood = local_neighbourhood*(local_neighbourhood>-2) - 1*(local_neighbourhood==-2)
#
#     # features = vicinity_index,tuple(local_neighbourhood)
#     # moves = self.vicinity_set[vicinity_index].actions[:-2] * np.logical_or(local_neighbourhood==0, local_neighbourhood==2)
#
#     features = (vicinity_index, tuple(local_neighbourhood), b_indicator) #bomb-indicator added as an additional feature
#
#     m_indicator = np.logical_or(local_neighbourhood==0, local_neighbourhood==2) #boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local enviroment
#
#     # allowed_actions = self.vicinity_set[vicinity_index].actions * ([m_indicator]+[True]+[b_indicator]) #boolena indicator for allowed actions: movements+WAIT+BOMB
#     allowed_actions = np.array(self.vicinity_set[vicinity_index].actions)[np.append(m_indicator, [True] + [b_indicator])]
#         # self.vicinity_set[vicinity_index].actions -> list [0,1,2,3,4,5]
#         # m_indicator -> np.array[True,False,True,True]
#         # allowed_actions is subset of a numpy array of [0,1,2,3,4,5]
#     return features, allowed_actions


# check if there is an available escape route
# xmin, xmax, ymin, ymax = vicinity_info[vicinity_index].explosion_range
# escape_route = []




def valid_boundary(x,min=0,max=16):
    '''return valid coordinate index for numpy array'''
    if x<min:
        return min
    if x>max:
        return max
    else:
        return x.astype(int)



