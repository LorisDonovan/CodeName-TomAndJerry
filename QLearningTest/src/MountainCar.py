import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000

SHOW_EVERY = 1000
STATS_EVERY = 200

start_qtable = f"C:\_My Files\Python\QLearningTest\QLearningTest\q_tables saved\q_tables Completed (1)\{24990}-qtable.npy"
#start_qtable = None

DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)	#separating the OS into 40 chunks
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#Exploration settings
if start_qtable is None:
	epsilon = 1
else:
	epsilon = 0.01

max_epsilon = 1
min_epsilon = 0.01
exploration_decay_rate = 0.001

if start_qtable is None:
	q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))	# 40x40x3 table
else:
	q_table = np.load(start_qtable)

ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg': [], 'min':[], 'max': []}

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
	episode_reward = 0
	if episode % SHOW_EVERY == 0:
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:
		# Exploration-exploitation:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		
		# Take new action:
		new_state, reward, done, _ = env.step(action)	#returns continuous states, need to be converted into discrete
		episode_reward += reward
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()
		
		 # Update Q-table:
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action, )]
			# Q-Learning equation:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q
		elif new_state[0] >= env.goal_position:
			print(f"Made it in episode: {episode}")
			q_table[discrete_state + (action, )] = 0

		# Set new state:
		discrete_state = new_discrete_state

	# Exploration rate decay
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-exploration_decay_rate*episode)

	ep_rewards.append(episode_reward)

	if episode % 10 == 0:
		np.save(f"C:\_My Files\Python\QLearningTest\QLearningTest\qtables\{episode}-qtable.npy", q_table)

	if not episode % STATS_EVERY:
		average_reward = sum(ep_rewards[-STATS_EVERY:])/len(ep_rewards[-STATS_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
		#print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')


env.close()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = "avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = "min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = "max")
plt.legend(loc=4)
plt.grid(True)
plt.show()
