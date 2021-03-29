import numpy as np
import pickle
from itertools import combinations, permutations, product
import matplotlib.pyplot as plt


# create dictionary to get state info from index for Q-table:

state_to_index = dict()
index_to_state = dict()
counter = 0
for i,v in enumerate(vicinity_set):
    neighbours = tuple(product([-1, 0, 1, 2, 3], repeat=len(v.neighbours)))
    nudge = tuple(product([-1, 0, 1], repeat=2))
    for n1,n2 in tuple(product(neighbours,nudge)):
        state = (i,n1,n2,True)
        state_to_index[state] = counter
        index_to_state[counter] = state

        state = (i,n1,n2, False)
        state_to_index[state] = counter+1
        index_to_state[counter+1] = state

        counter +=2

len(state_to_index) #2550 -> #22950
len(index_to_state) #2550 *9

with open("./agent_code/agent_SUN/state_to_index.pk", "wb") as file:
    pickle.dump(state_to_index, file)

with open("./agent_code/agent_SUN/index_to_state.pk", "wb") as file:
    pickle.dump(index_to_state, file)


with open("./agent_code/agent_SUN/model/state_to_index.pk", "rb") as file:
    state_to_index = pickle.load(file)

with open("./agent_code/agent_SUN/model/index_to_state.pk", "rb") as file:
    index_to_state = pickle.load(file)

vicinity_types = np.loadtxt('./agent_code/agent_SUN/model/vicinity_types.csv', delimiter=',')
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



Q_save = 'agent_code/agent_SUN/model/Q_SUN.pk'
with open(Q_save, "rb") as file:
    Q = pickle.load(file)

with open(path+"Q_tracker_SUN.pk", "rb") as file:
    Q_tracker = pickle.load(file)

print(f"Q_tracker: max={Q_tracker.max()}, sum={Q_tracker.sum()}, distribution: {np.unique(Q_tracker, return_counts=True)}")
np.savetxt(path+'Q_tracker_1_10000.csv',Q_tracker, delimiter=',')
plt.hist(Q_tracker)


path = './agent_code/trainer_agent/model/'
'Q_tracker_trainer_1_10000_coins.pk'
tracker_save = path+'Q_tracker_trainer_1_20000_crates.pk'
'Q_trainer_1_10000_coins.pk'
Q_save = path+'Q_trainer_1_20000_crates.pk'


def display_value(Q,x,text):
    'find a value (=x) in table (Q or tracker), display indices and correpsonding state-action'
    # s_,a_ = np.where(Q==x)[0].item(),np.where(Q==x)[1].item() # if more than once --> ValueError: can only convert an array of size 1 to a Python scalar
    # print(f'{text} = {x} at Q[{s_,a_}], corresponding state {index_to_state[s_]}, action {ACTIONS[a_]}')

    c = np.where(Q==x)
    for i in range(len(c[0])):
        s_,a_ = c[0][i], c[1][i]
        print(f'{text} = {x} at Q[{s_, a_}], corresponding state {index_to_state[s_]}, action {ACTIONS[a_]}')
# display_value(Q,Q.max(),'Q-values: max')
# display_value(tracker,tracker.max(),'Tracker: max')

def display_update(Q, tracker, n):
    'find Q values where updates num > n, display indices in table and corresponding state-action'
    for value in Q[tracker >n]:
        idx, action = np.where(Q == value)
        print(f'For >{n} updates: Q = {value}, at Q{idx}{action}, corresponding state {[index_to_state[j] for j in idx]},action {[ACTIONS[k] for k in action]}')
# display_update(Q, tracker, 100000)

def run_diagnosis(Q_save,tracker_save,n_round):

    with open(Q_save, "rb") as file:
        Q = pickle.load(file)

    with open(tracker_save, "rb") as file:
        tracker = pickle.load(file)

    ### stats
    total = Q.size  # 137700 Total
    invalid = np.sum(Q == -np.Inf)  # 72315 Invalid
    not_updated = np.sum(Q == 0)  # 62800 not updated
    positive = np.sum(Q > 0)  # 2584 # positive states
    negative = np.sum(Q < 0) - invalid  # 72316 # negative states
    print(f'Q-stats: #{positive+negative} updated ({(positive+negative)/total*100:.1f}%), '
          f'#{not_updated} not updated ({not_updated/total*100:.1f}%), #{invalid} invalid ({invalid/total*100:.1f}%)')
    # print(f'Q-values: histogram {np.histogram(Q[Q>-np.Inf],bins = 10)}')

    display_value(Q, Q.max(), 'Q-max')
    minQ = Q[Q>-np.Inf].min()
    if minQ < 0:
        display_value(Q, minQ, 'Q-min')
    else:
        print(f'Q-min = {minQ}')
    display_value(tracker, tracker.max(), 'Tracker-max')
    display_update(Q, tracker, 100000)


    ### plots % of updates according to cell distribution

    # a list of total updates for each cell type
    updates = np.zeros(11)
    for k,v in state_to_index.items():
        updates[k[0]] += tracker[v].sum()
    # array([1472744., 1089521., 1090903.,  137169.,  166873.,  138227.,  167647.,   19063.,   20041.,   20885.,   20053.])
    updated = updates/tracker.sum()
    print(f'Updates by type: {updated}')
    # Update distribution
    # array([0.33909769, 0.25086102, 0.25117922, 0.03158301, 0.03842233, 0.03182662, 0.03860054, 0.00438923, 0.00461442, 0.00480875, 0.00461718])


    # cell type distribution on gameboard
    np.unique(vicinity_types,return_counts=True)[1] #array([113,  36,  56,  56,   6,   6,   6,   6,   1,   1,   1,   1]
    original = np.unique(vicinity_types,return_counts=True)[1]/vicinity_types.size
    # array([0.39100346, 0.12456747, 0.19377163, 0.19377163, 0.02076125, 0.02076125, 0.02076125, 0.02076125, 0.00346021, 0.00346021, 0.00346021, 0.00346021])

    plt.plot(original, label='original')
    plt.plot(updated,label='updated')
    plt.legend()
    plt.title(f"Cell type distribution ({n_round} rounds, {tracker.sum()} updates)")
    plt.show()

# run_diagnosis(Q_save,tracker_save,n_round=10000)



# DONE: add two extra bits to state to represent x:left(-1)/right(+1)/no(0), y:up(-1)/down(+1)/no(0) towards nearest target -- nudge
#  - function find nearest target [DONE]
#  - generate new dict{} state->index, index->state (should be 2550 * 9(3^2)) [DONE]
#  - auxiliary award to moving towards nearest target
#  - train a single trainer without crates 10,000 round Q_trainer [DONE]
#  - agent play some games [DONE] -- not great with eps = 0.2; not really heading towards targets, doodling around in its own neighbourhood
#  - train agent_SUN based on Q_trainer same condition (no crates) 1,000 round with esp_train = 0.5 and hopefully will go far [DONE]

### problem encountered: agent still 'trapped' in a corner and not moving towards targets
### ideas to implement
    # added a booster to 'teleport' to another neighbourhood - DONE
    # nudging is not really working - change reward / add target tracking? DONE
    # diagnose Q and Q-tracker, check for convergence DONE

### Training procedure for coin only [alpha = 0.5, gamma = 0.5, crates density = 0]:
    # train an explorer (rule-based just going through all possible tiles on game board - 1000 round >>>Q_1000
    # based on experience of the explorer (update Q from above step), train agent_SUN with eps_train = 1 (random exploration amongst allowed actions with preferred nudge)
    # train agent_SUN for increment of 100 round, saving Q and Q-tracker at 100, 200, 300, 400 and 500 for further diagnoalsis >>>Q_1500
    # play game with eps = 0.1 is better than 0
    # OBSERVATION: seems to be effective at collecting coins, min. 67 steps for 9 coins, first achieved at 200 round; max c. 120 steps

### Training procedure for single agent with crates [alpha = 0.5, gamma = 0.5, crates density = 0.75 as per game setting]:
    # start with Q_1500, train tainer_agent (rule-based) - 1000 round >>>Q_2500
    #

# TODO:
    # change alpha, gamma and epsilon
    # figure out the optimal minimal training required
    # add randomness to rule-based trainer?
    # how to track convergence?

# Set up an explorer agent to learn about the grid

def compare(versions:list,path='./agent_code/agent_SUN/model/'):
    for version in versions:
        # path = './agent_code/agent_SUN/model/'
        # version = '_1000_coins'
        print(f'\n >>> version{version}')
        tracker_save = f'{path}Q{version}_tracker.pk' #_explorer_1000_coins
        Q_save = f'{path}Q{version}.pk'
        run_diagnosis(Q_save,tracker_save,n_round=version)

compare(['_1000_coins','_1100_coins','_1200_coins','_1300_coins','_1400_coins','_1500_coins'])
compare(versions=[''],path='./agent_code/explorer/model/')
compare(versions=[''],path='./agent_code/trainer/model/')

def nudge_direction(nudge):
    flag = np.array([nudge[1]==-1,nudge[0]==1,nudge[1]==1,nudge[0]==-1,nudge[0]==nudge[1]==0])
    action_idx = np.arange(5)[flag].astype(int)
    return action_idx


### current training strategy
# train agent with coins only directly with eps_train = 1, i.e. always explore, can effectively get all coins
# train trainer with crates - after 5000 round, still kills itself - does not know how to escape its own bomb
# suspected that reward/penalty for towards/away from nearest targets might have mismatch between trainer (dead-end?) and agent
# therefore changed training strategy - muted towards/away from reward and retrain with trainer on top of 1000 round of coins
# trained trainer for 1000, 2000, and 5000 round, still no improvement when agent plays with eps 0.1 or 0.2
# start training agent with eps_train = 0.5
# current label '_coins_SUN_1000_crates_trainer_5000_noNudgeReward_SUN_1000_eps0.5'


compare(versions=['_100_eps_1','_500_eps_1','_1000_eps_1'],path='./agent_code/agent_SUN/model/coins/')

def read_score_card(versions=[i for i in range(100,1100,100)],
                    path='./agent_code/agent_SUN/model/coins/',
                    pre = 'score_card_10_rounds_',post = '_eps_1.csv'):
    avg_steps = []
    avg_score = []
    for version in versions:
        filename = f"{path}{pre}{version}{post}"
        a = np.loadtxt(filename,delimiter=',')
        steps,score = a[1:,1:].mean(axis=0)
        avg_score.append(score)
        avg_steps.append(steps)
    return avg_steps, avg_score



# coin collection analysis visualization

avg_steps, avg_scores = read_score_card([i for i in range(100,1100,100)],path='./agent_code/agent_SUN/model/coins/', pre = 'score_card_10_rounds_',post = '_eps_1.csv')
a = np.loadtxt(path+'score_card_10_rounds_Q_2000_eps_0.15.csv', delimiter=',')
steps, score = a[1:, 1:].mean(axis=0)
avg_steps.append(steps)
rounds = [i for i in range(100,1100,100)]+[2000]
plt.plot(rounds,avg_steps,'bo',linestyle='--',label='no reward for movement')
plt.title("Avg Num of Steps to Complete Coins Collection \n(Sample of 10 Rounds)")

avg_steps, avg_scores = read_score_card(versions=[i for i in range(100,1100,100)], path='./agent_code/agent_SUN/model/coins_rewardMovement/',pre = 'score_card_10_rounds_',post = '_0.5_0.5_1.csv')
a = np.loadtxt('./agent_code/agent_SUN/model/coins_rewardMovement/score_card_10_rounds_2000_0.5_0.5_0.15.csv', delimiter=',')
steps, score = a[1:, 1:].mean(axis=0)
avg_steps.append(steps)
plt.plot(rounds,avg_steps,'ro',linestyle='--',label='with reward for movement')
plt.legend()
plt.ylim(50,270)
plt.ylabel("Avg. Num of steps per round")
plt.xlabel('Num of training round')
plt.savefig('./agent_code/agent_SUN/figure/coin_collection_analysis.png')
plt.show()

# Q updated based on trainer; crates density = 25%, esp =0.2
rounds =[i for i in range(100,1000,100)]+[i for i in range(1000,11000,1000)]
avg_steps, avg_scores = read_score_card(rounds,path='./agent_code/agent_SUN/model/crates/trainer/esp_0.2/', pre = 'score_card_10_rounds_',post = '_eps_0.5.csv')
plt.plot(rounds,avg_steps)
plt.show()

plt.plot(rounds,avg_scores)




# reconsidering of adding escape route indicator
agent = [None,None,True,(10,10)]
x, y = agent[-1]
b_indicator = agent[2]
vicinity_index = 0 #(self.vicinity_type[y, x]).astype(int)
vicinity = vicinity_set[vicinity_index]  # Vicinity = namedtuple('Vicinity', ['neighbours', 'explosion_range', 'actions'])
neighbours = [(x + ij[0], y + ij[1]) for ij in vicinity.neighbours]  # get global coordinates of neighbours

local_neighbourhood = np.array([0,1,2,0]) #np.array([self.treasure_map[y_, x_] for (x_, y_) in neighbours])  # get identities of neighbours from treasure map

# boolen indicator for allowed movements [UP,RIGHT,DOWN,LEFT] based on local environment
m_indicator = np.logical_or(local_neighbourhood == 0,local_neighbourhood == 2)  # np.array[True,False] corresponding to vicinity.actions (already restricted)

# NEWLY ADDED 2021.03.26
# Check explosion range for escape route; if exists then bomb, otherwise don't.
# get global coordinates of potential escape routes based on 'safe' neighbours only
escape_routes = []
for (sn_xy, sn_ij,m) in zip(np.array(neighbours),np.array(vicinity.neighbours),m_indicator):  # global and local coordinates of 'safe' neighbours
    if not m:
        escape_routes.append(0)
    else:
        route_1 = [sn_xy + sn_ij*i for i in range(1,4)]
        route_2 = [sn_xy + sn_ij, sn_xy + sn_ij + 1*np.array(sn_ij==0)]
        route_3 = [sn_xy + sn_ij, sn_xy + sn_ij + (-1)*np.array(sn_ij==0)]

        safe_route_counter = 0
        for route in [route_1,route_2,route_3]:
            route_ = np.array([0,0,2,0]) #np.array([self.treasure_map[y_, x_] for (x_, y_) in route])
            safe = np.all(np.logical_or(route_ == 0, route_ == 2))
            safe_route_counter += safe
        escape_routes.append(safe_route_counter)

np.argmax(escape_routes)
max(escape_routes)
np.where(escape_routes==max(escape_routes))
random.choice([0])

def validify_boundary(x,min=0,max=16):
    if x<min:
        return min
    if x>max:
        return max
    else:
        return x

validify_boundary(16)


compare(versions=['_100_eps_0.2','_500_eps_0.2','_1000_eps_0.2'],path='./agent_code/trainer/model/crates/gamma_0/')
compare(versions=['_100_eps_0.5','_500_eps_0.5','_1000_eps_0.5'],path='./agent_code/trainer/model/crates/crates_0.5_0.5/')
compare(versions=['_1000_eps_0.5','_5000_eps_0.5','_10000_eps_0.5'],path='./agent_code/trainer/model/crates/crates_0.5_0.5/')

# training by observing trainers play: no negative penalty as trainer does not make mistakes and never dies in single player model, as a result Q table is all positive

# training with smaller board 9x9 and less crates, based on previous 1000 rounds of training of collecting coins
compare(versions=['_coins_1000','_100','_500','_1000'],path='./agent_code/agent_SUN/model/crates_9x9_0.50/')
compare(versions=['_2000','_4000','_6000'],path='./agent_code/agent_SUN/model/crates_9x9_0.50/')
# c. 1300 updates per round c. 1% of Q states updated

# next --> need to train trainers to play with each other and learn the game dynamic
    # change settings.py back to game environement
    # change environment.py coins distribtuion in enviroments back to default settings
    # update in agent callback.py
        # path = 'model/game/'
        # version = 'init_4_trainer_1000' #got killed -100
    # > python main.py play --agents trainer trainer trainer trainer --train 4 --n-round 10000 --no-gui

    # then train for killing other agent
    # > python main.py play --agents trainer trainer peaceful_agent random_agent --train 2 --n-round 5000 --no-gui

compare(versions=['_100','_500','_1000','_5000','_10000'],path='./agent_code/trainer/model/game/4_trainers/')
compare(versions=['_100','_200','_300','_400','_500'],path='./agent_code/trainer/model/game/')

# think of a way to combine all Q-tables, maybe weighted by 1/(rounds/1000); maybe use regression???







### 9x9 crates density 0.50 analysis visualization
compare(versions=['_100','_500','_1000','_5000','_10000'],path='./agent_code/Agent_SUN/model/crates_9x9_0.50/')

rounds = [i for i in range(100,1100,100)]+[i for i in range(1000,11000,1000)]
avg_steps, avg_scores = read_score_card(rounds,path='./agent_code/agent_SUN/model/crates_9x9_0.50/kill_100_beforeNullBomb/', pre = 'score_card_10_rounds_',post = '.csv')

fig, ax = plt.subplots(2,1,sharex=True)

ax[0].plot(rounds,avg_steps,'bo',linestyle='--',label='')
ax[0].set_title("Performance in Task 2 (grid 9x9, crate density 0.5)\n(Sample of 10 Rounds)")
ax[0].set_ylabel("Num of steps survived")
ax[0].set_xlabel('Num of training round')

ax[1].plot(rounds,avg_scores,'ro',linestyle='--',label='')
ax[1].set_ylabel("Score")
ax[1].set_xlabel('Num of training round')

# plt.legend()
# ax[0].ylim(50,270)
fig.savefig('./agent_code/agent_SUN/figure/crates_9x9_analysis.png')
plt.show()



### 9x9 crates density 0.50 analysis visualization - Penalty for dropping bomb without destruction

compare(versions=['_100','_500','_1000','_5000','_10000'],path='./agent_code/Agent_SUN/model/crates_9x9_0.50/kill_100_NullBomb_0.5/')

rounds = [i for i in range(100,1100,100)]+[i for i in range(1000,11000,1000)]
avg_steps, avg_scores = read_score_card(rounds,path='./agent_code/agent_SUN/model/crates_9x9_0.50/kill_100_NullBomb_0.5/', pre = 'score_card_10_rounds_',post = '.csv')

fig, ax = plt.subplots(2,1,sharex=True)

ax[0].plot(rounds,avg_steps,'bo',linestyle='--',label='')
ax[0].set_title("Performance in Task 2 (grid 9x9, crate density 0.5)\n(Sample of 10 Rounds)")
ax[0].set_ylabel("Num of steps survived")
ax[0].set_xlabel('Num of training round')

ax[1].plot(rounds,avg_scores,'ro',linestyle='--',label='')
ax[1].set_ylabel("Score")
ax[1].set_xlabel('Num of training round')

# plt.legend()
# ax[0].ylim(50,270)
fig.savefig('./agent_code/agent_SUN/figure/crates_9x9_analysis.png')
plt.show()




### trainers game observation

compare(versions=['init_4_trainer_10000','init_2_trainers_random_peaceful_5000'],path='./agent_code/trainer/model/game/')

def combine(weights = [],versions=['init_4_trainer_10000','init_2_trainers_random_peaceful_5000'],path='./agent_code/trainer/model/game/'):
    Q_combined = np.zeros((22950, 6))
    for w, version in zip(weights,versions):
        # tracker_save = f'{path}Q{version}_tracker.pk'
        Q_save = f'{path}Q{version}.pk'
        with open(Q_save, "rb") as file:
            Q = pickle.load(file)
        # with open(tracker_save, "rb") as file:
        #     tracker = pickle.load(file)
        # normalize Q
        # Q_norm = Q[np.where(Q>=0)]/Q.max() + Q[Q<0]/Q[Q!=-np.Inf].min()
        Q_min = Q[Q!=-np.Inf].min()
        if Q_min < 0:
            Q_norm = np.where(Q>0, Q/Q.max(), Q/-Q_min)
        else:
            Q_norm = Q/Q.max()

        Q_combined += Q_norm * w

        total = Q.size  # 137700 Total (fixed)
        invalid = np.sum(Q == -np.Inf)  # 72315 Invalid (fixed)
        not_updated = np.sum(Q == 0)  # 62800 not updated
        positive = np.sum(Q > 0)  # 2584 # positive states
        negative = np.sum(Q < 0) - invalid  # 72316 # negative states
        print(f'>> version_{version}: \n\t Q-max={Q.max():.1f},Q-min={Q_min:.1f}, Normalized = [{Q_norm.max(),Q_norm[Q_norm!=-np.Inf].min()}], '
              f'\n\t Q-stats: #{positive + negative} updated ({(positive + negative) / total * 100:.1f}%), '
              f'#{not_updated} no update ({not_updated / total * 100:.1f}%)')

    Q_ = Q_combined / sum(weights)
    Q_min = Q_[Q_ != -np.Inf].min()
    if Q_min < 0:
        Q_norm = np.where(Q_ > 0, Q_ / Q_.max(), Q_ / -Q_min)
    else:
        Q_norm = Q_ / Q_.max()

    not_updated = np.sum(Q_ == 0)  # 62800 not updated
    positive = np.sum(Q_ > 0)  # 2584 # positive states
    negative = np.sum(Q_ < 0) - invalid  # 72316 # negative states
    print(f'>> COMBINED with weights = {weights}: \n\t Q-max={Q_.max():.1f},Q-min={Q_min:.1f}, Normalized = [{Q_norm.max(), Q_norm[Q_norm != -np.Inf].min()}], '
        f'\n\t Q-stats: #{positive + negative} updated ({(positive + negative) / total * 100:.1f}%), '
        f'#{not_updated} no update ({not_updated / total * 100:.1f}%)')

    return Q_norm

Q_ = combine(weights = [0.5,0.5],versions=['init_4_trainer_10000','init_2_trainers_random_peaceful_5000'],path='./agent_code/trainer/model/game/')


Q_ = combine(weights = [1,1,1,1],versions=['_coins_900','_crates_6000','init_4_trainer_10000','init_2_trainers_random_peaceful_5000'],path='./agent_code/trainer/model/game/')

Q_ = combine(weights = [1,1],versions=['_coins_900','_crates_6000'],path='./agent_code/trainer/model/game/')
with open('./agent_code/agent_SUN/model/game/alone/Q_combined.pk', "wb") as file:
    pickle.dump(Q_, file)


compare(versions=['_1000_eps_0.5','_5000_eps_0.5','_10000_eps_0.5'],path='./agent_code/agent_SUN/model/game/alone/eps_0.5/')
compare(versions=['_combined','_CURRENT'],path='./agent_code/agent_SUN/model/game/alone/')

with open('./agent_code/agent_SUN/model/game/alone/Q_CURRENT.pk', "rb") as file:
    Q = pickle.load(file)


Q_ = combine(weights = [1,1,1,1],versions=['_coins_crates_games_50k','_trainer_crates_10k','init_4_trainer_10000','init_2_trainers_random_peaceful_5000'],path='./agent_code/agent_SUN/model/combine/')
with open('./agent_code/agent_SUN/model/combine/Q_combined.pk', "wb") as file:
    pickle.dump(Q_, file)





# calculates average score per step for each game then take average of 10 games
def read_score_card_(versions=[i for i in range(100,1100,100)],
                    path='./agent_code/agent_SUN/model/coins/',
                    pre = 'score_card_10_rounds_',post = '_eps_1.csv'):
    avg_steps = []
    avg_score = []
    avg_score_per_step = []
    for version in versions:
        filename = f"{path}{pre}{version}{post}"
        a = np.loadtxt(filename,delimiter=',')
        steps,score = a[1:,1:].mean(axis=0)
        score_per_step = np.mean((a[1:,2]) / a[1:,1])
        avg_score.append(score)
        avg_steps.append(steps)
        avg_score_per_step.append(score_per_step)
    return avg_steps, avg_score, avg_score_per_step

# Performance of Coin Collection (Task 1)
avg_steps, avg_scores, avg_score_per_step = read_score_card_([i for i in range(100,1100,100)],path='./agent_code/agent_SUN/model/coins/', pre = 'score_card_10_rounds_',post = '_eps_1.csv')
# score_per_step = (np.array(avg_scores)+1) / np.array(avg_steps)
# need to +1 here since the score card does not record the last coin collected.
a = np.loadtxt(path+'score_card_10_rounds_Q_2000_eps_0.15.csv', delimiter=',')
# steps, score = a[1:, 1:].mean(axis=0) # first row are just all zeros
score_per_step = np.mean((a[1:, 2] + 1) / a[1:, 1])
avg_score_per_step = np.append(avg_score_per_step, score_per_step)
rounds = [i for i in range(100,1100,100)]+[2000]

peaceful = np.loadtxt('./agent_code/peaceful_agent/score_card_10_random3.csv', delimiter=',')
peaceful_score_per_step = np.mean((peaceful[1:, 2] + 1) / peaceful[1:, 1])
np.mean(peaceful[1:, 1]),np.mean((peaceful[1:, 2] + 1))

rulebased = np.loadtxt('./agent_code/trainer/model/coins/score_card_10_trainer.csv', delimiter=',')
rule_score_per_step = np.mean((rulebased[1:, 2] + 1) / rulebased[1:, 1])
np.mean(rulebased[1:, 1]),np.mean((rulebased[1:, 2] + 1)),rule_score_per_step

plt.plot(avg_score_per_step,'bo',linestyle='--',label='Agent SUN')
plt.axhline(rule_score_per_step,color='orange',linestyle='--',label='Rule-based Agent')
plt.axhline(peaceful_score_per_step,color='green',linestyle='--',label='Peaceful-random Agent')

plt.title("Task 1: Performance in Coin Collection \n(Based on Samples of 10 Rounds)")
plt.legend()
plt.ylabel("Avg. score per step")
plt.ylim(0,0.2)
plt.xlabel('Num of rounds in training')
plt.xticks(ticks=[i for i in range(len(rounds))],labels = rounds)
plt.savefig('./agent_code/agent_SUN/figure/coin_performance_benchmark.png')
plt.show()

plt.close()



# performance of task 2
# 9x9
rounds = [i for i in range(100,1100,100)]+ [i for i in range(1000,11000,1000)]
avg_steps, avg_scores, avg_score_per_step = read_score_card_(rounds,path='./agent_code/agent_SUN/model/crates_9x9_0.50/kill_100_NullBomb_0.5/', pre = 'score_card_10_rounds_',post = '.csv')
plt.plot(avg_score_per_step,'bo',linestyle='--',label='Agent SUN')
plt.show()



plt.close()
# game alone 50000
rounds = [i for i in range(1000,51000,1000)]
# avg_steps, avg_scores, avg_score_per_step = read_score_card_(rounds,path='./agent_code/agent_SUN/model/game/alone/eps_0.5/', pre = 'score_card_10_rounds_',post = '_esp_0.5.csv')
avg_steps, avg_scores, avg_score_per_step = read_score_card_(rounds,path='./agent_code/agent_SUN/model/game/alone/eps_0.5/', pre = 'score_card_10_rounds_',post = '_eps_0.5_eps0.1.csv')
plt.plot(avg_score_per_step,'bo',linestyle='--',label='Agent SUN')

random = np.loadtxt('./agent_code/random_agent/score_card_10_random.csv', delimiter=',')
random_score_per_step = np.mean((random[1:, 2]) / peaceful[1:, 1])
np.mean(random[1:, 1]),np.mean((random[1:, 2]))

trainer = np.loadtxt('./agent_code/agent_SUN/model/combine/score_card_10_rounds_50k_trainer_eps0.1.csv', delimiter=',')
trainer_score_per_step = np.mean((trainer[1:, 2]) / trainer[1:, 1])
np.mean(trainer[1:, 1]),np.mean((trainer[1:, 2])),trainer_score_per_step

# combo = np.loadtxt('./agent_code/agent_SUN/model/combine/score_card_10_rounds_coins_crates_games_50k_eps0.05.csv', delimiter=',')
# combo_score_per_step = np.mean((combo[1:, 2]) / combo[1:, 1])
# np.mean(combo[1:, 1]),np.mean((combo[1:, 2])),combo_score_per_step

rulebased = np.loadtxt('./agent_code/trainer/model/game/score_card_10_trainer.csv', delimiter=',')
rule_score_per_step = np.mean((rulebased[1:, 2]) / rulebased[1:, 1])
np.mean(rulebased[1:, 1]),np.mean((rulebased[1:, 2])),rule_score_per_step

plt.axhline(rule_score_per_step,color='orange',linestyle='--',label='Rule-based Agent')
plt.axhline(trainer_score_per_step,color='red',linestyle='--',label='Imitating Trainer')
plt.axhline(random_score_per_step,color='green',linestyle='--',label='Random Agent')
# plt.axhline(combo_score_per_step,color='red',linestyle='--',label='Combo Agent')

plt.title("Task 2: Performance in Bombing Crates \n(Based on Samples of 10 Rounds, epsilon=0.1)")
plt.legend(loc='upper right')
plt.ylabel("Avg. score per step")
# plt.ylim(0,0.2)
plt.xlabel('Num of rounds in training (\'000)')
# plt.xticks(ticks=[5*(i+1) for i in range(9)],labels = rounds[9::5])
plt.savefig('./agent_code/agent_SUN/figure/crates_performance_benchmark_0.1_.png')
plt.show()


Q_ = combine(weights = [1,1],versions=['init_4_trainer_10000','init_2_trainers_random_peaceful_5000'],path='./agent_code/agent_SUN/model/combine/')
with open('./agent_code/agent_SUN/model/combine/Q_trainer.pk', "wb") as file:
    pickle.dump(Q_, file)

Q_ = combine(weights = [1,1],versions=['_trainer','_50000_alone'],path='./agent_code/agent_SUN/model/combine/')
with open('./agent_code/agent_SUN/model/combine/Q_50k_trainer.pk', "wb") as file:
    pickle.dump(Q_, file)



compare(versions=['_1000_eps_0.5','_5000_eps_0.5','_10000_eps_0.5','_50000_eps_0.5'],path='./agent_code/agent_SUN/model/game/alone/eps_0.5/')
