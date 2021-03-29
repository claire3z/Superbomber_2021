import logging

from fallbacks import pygame

# Game properties - Default settings
COLS = 17
ROWS = 17
CRATE_DENSITY = 0.75
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400




# Settings for Task 1: On a game board without any crates or opponents, collect a number of revealed coins as quickly as possible.
# The agent should learn how to navigate the board efficiently.
# CRATE_DENSITY = 0 # for coin_hunter (active bombing crates) to compete with coin_collector (passively collecting)

# Settings for Task 2:
# CRATE_DENSITY = 0.25 # to train for effective escape from own bombing

# COLS = 9
# ROWS = 9
# CRATE_DENSITY = 0.5
# MAX_AGENTS = 4
# MAX_STEPS = 400

# TODO Remember to change enviroment coins setting as well!!!

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2

# Rules for agents
TIMEOUT = 5
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
LOG_GAME = logging.INFO
LOG_AGENT_WRAPPER = logging.DEBUG
LOG_AGENT_CODE = logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
