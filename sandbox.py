'''This script is to play a game manually - need to restart after each game'''

import sys
import threading
from argparse import ArgumentParser
from time import sleep, time

import settings as s
from environment import BombeRLeWorld, GenericWorld
from fallbacks import pygame, tqdm, LOADED_PYGAME
from replay import ReplayWorld


# Function to run the game logic in a separate thread
def game_logic(world: GenericWorld, user_inputs, args):
    last_update = time()
    while True:
        now = time()
        if args.turn_based and len(user_inputs) == 0:
            sleep(0.1)
            continue
        elif world.gui is not None and (now - last_update < args.update_interval):
            sleep(args.update_interval - (now - last_update))
            continue

        last_update = now
        if world.running:
            world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')


class Args():
    def __init__(self):
        self.train = 0
        self.continue_without_training = False #-- default
        self.my_agent = None
        self.agents = ["coin_hunter","coin_collector","user_agent"] #["rule_based_agent"] * s.MAX_AGENTS
        self.n_rounds = 1
        self.save_replay = False
        self.no_gui = False
        self.fps = 15
        self.turn_based = False
        self.update_interval = 0.5
        self.make_video = False
        self.log_dir = 'logs'
args = Args()

has_gui = not args.no_gui
if has_gui:
    if not LOADED_PYGAME:
        raise ValueError("pygame could not loaded, cannot run with GUI")
    pygame.init()

# Initialize environment and agents

agents = []
if args.train == 0 and not args.continue_without_training:
    args.continue_without_training = True
if args.my_agent:
    agents.append((args.my_agent, len(agents) < args.train))
    args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
for agent_name in args.agents:
    agents.append((agent_name, len(agents) < args.train))

world = BombeRLeWorld(args, agents)

user_inputs = []

# Start game logic thread
t = threading.Thread(target=game_logic, args=(world, user_inputs, args), name="Game Logic")
t.daemon = True
t.start()

# Run one or more games
for _ in tqdm(range(args.n_rounds)):
    if not world.running:
        world.ready_for_restart_flag.wait()
        world.ready_for_restart_flag.clear()
        world.new_round()

    # First render
    if has_gui:
        world.render()
        pygame.display.flip()

    round_finished = False
    last_frame = time()
    user_inputs.clear()

    # Main game loop
    while not round_finished:
        if has_gui:
            # Grab GUI events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if world.running:
                        world.end_round()
                    world.end()
                    # return
                elif event.type == pygame.KEYDOWN:
                    key_pressed = event.key
                    if key_pressed in (pygame.K_q, pygame.K_ESCAPE):
                        world.end_round()
                    if not world.running:
                        round_finished = True
                    # Convert keyboard input into actions
                    if s.INPUT_MAP.get(key_pressed):
                        if args.turn_based:
                            user_inputs.clear()
                        user_inputs.append(s.INPUT_MAP.get(key_pressed))

            # Render only once in a while
            if time() - last_frame >= 1 / args.fps:
                world.render()
                pygame.display.flip()
                last_frame = time()
            else:
                sleep_time = 1 / args.fps - (time() - last_frame)
                if sleep_time > 0:
                    sleep(sleep_time)
        elif not world.running:
            round_finished = True
        else:
            # Non-gui mode, check for round end in 1ms
            sleep(0.001)

world.end()
