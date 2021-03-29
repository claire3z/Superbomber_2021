import numpy as np

path = ''

def setup(self):
    np.random.seed()
    ####### for record keeping and experiment analysis REMOVE BEFORE COMPETITION TODO
    self.round = 0
    self.step = 0
    self.score = 0
    self.record = []
    #######

# def act(agent, game_state: dict):
#     agent.logger.info('Pick action at random, but no bombs.')
#     return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])

def act(self, game_state: dict):
    self.logger.info('Pick action at random, but no bombs.')

    ###### for record keeping and experiment analysis REMOVE BEFORE TRAINING TODO python main.py play --agents agent_SUN --n-round 11 --no-gui

    if game_state['round'] == self.round+1:
        self.record.append([self.round, self.step, self.score])
        self.round = game_state['round']
    else:
        self.step = game_state['step']
        self.score = game_state['self'][1]

    if len(self.record) == 11:
        np.savetxt(f'{path}score_card_{self.round-1}_random.csv', self.record, delimiter=',')

    ######


    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
