from collections import deque
from random import shuffle
import numpy as np
import os
import pickle
import random
from collections import namedtuple
import settings as s

##############################################################################

path = 'C:/Users/clair/Desktop/WS2020/FML/bomberman_rl/agent_code/agent_SUN/'

SAFE = 99
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
Q_save = path + 'Q_SUN.pk'

Vicinity = namedtuple('Vicinity', ['neighbours', 'explosion_range', 'actions']) # explosion range [x.min, x.max,y.min, y.max], ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
vicinity_type = np.loadtxt('./agent_code/agent_SUN/vicinity_types.csv', delimiter=',')

# This need to be pre-generated depending on state features
with open("./agent_code/agent_SUN/state_to_index.pk", "rb") as file:
    state_to_index = pickle.load(file)

# NEW state = (vicinity_type,(neighbours),b_indicator)
def initialize_Q():
    Q = np.zeros([len(state_to_index),len(ACTIONS)])-np.Inf
    for k,v in state_to_index.items():
        # k = (vicinity_type,(neighbours),b_indicator)
        actions = np.array(vicinity_set[k[0]].actions) #all legal moves defined by vicinity type
        # customized by actual surrounding situation, only truly allowed if free_tile(0) OR coin(2)
        mask = np.logical_or(np.array(k[1]) == 0,np.array(k[1]) == 2)
        mask = np.append(mask, [True]+[k[-1]]) # True for WAIT; k[-1] is BOMB_indicator
        valid_actions = actions[mask]
        Q[v][valid_actions] = 0
    # np.sum(Q == -np.Inf) #15300 --> #8035 = 3380*2 + 2550/2
    return Q

##############################################################################


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

    ##############################################################################
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

    ##############################################################################


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0


def act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] <= 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    # dead_ends: max crates to blow up
    dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a


######################################################################################################################

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
    field = game_state['field']

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

    self.minefield = np.maximum(explosion_map * -1 + 1 , bombs_map * -1 - 1)

    # combine minefield with field_with_, minefield overwrites other objects
    self.global_map = ((self.minefield < -4) * self.treasure_map + (self.minefield >= -4) * self.minefield)

    # my own agent's coordinates
    x, y = game_state['self'][-1]

    # update my position in global map
    self.global_map[y,x] = 7



def infer_local_features(self, agent):
    """agent: A tuple (name, score, b_indicator,(x, y)) describing an agent: game_state['self] or game_state['others'][i]
    return: a tuple (feature, allowed actions) where feature -> tuple(vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """
    x, y = agent[-1]
    b_indicator = agent[2]
    vicinity_index = (self.vicinity_type[y, x]).astype(int)
    neighbours = [(x + ij[0], y + ij[1]) for ij in self.vicinity_set[vicinity_index].neighbours]

    local_neighbourhood = np.array([self.treasure_map[j, i] for (i, j) in neighbours])

    features = (vicinity_index, tuple(local_neighbourhood), b_indicator) #bomb-indicator added as an additional feature

    m_indicator = np.logical_or(local_neighbourhood==0, local_neighbourhood==2) #boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local enviroment
    allowed_actions = np.array(self.vicinity_set[vicinity_index].actions)[np.append(m_indicator, [True] + [b_indicator])]

    return features, allowed_actions



def state_to_features(self, game_state: dict) -> tuple:
    """ Converts the game state to the input of your model, i.e. a feature vector.
    :param game_state:  A dictionary describing the current game board.
    :return: a tuple (feature, allowed actions) where feature = (vicinity_type,(neighbours),(b_indicator)), allowed actions ->list """

    if game_state is None:
        return None,None  # This needs to conform to return below

    create_global_map(self,game_state)
    my_agent = game_state['self']
    features, allowed_actions = infer_local_features(self, my_agent)

    return features, allowed_actions

