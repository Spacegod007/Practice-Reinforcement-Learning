import gym
import numpy as np
import matplotlib.pylab as plt


def main(envName):

    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.int))

    env = gym.make(envName)
    env.reset()

    learning_rate = 0.1
    discount = 0.95
    episodes = 2000

    show_every = 500

    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

    epsilon = 0.5
    start_epsilon_decaying = 1
    end_epsilon_decaying = episodes // 2

    epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

    q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))

    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

    for episode in range(episodes + 1):

        episode_reward = 0

        if (not episode % show_every != 0):
            print(episode)
            render = True
        else:
            render = False

        discrete_state = get_discrete_state(env.reset())

        done = False
        while not done:

            if (np.random.random() > epsilon):
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            observation, reward, done, info = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(observation)

            if (render):
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]

                new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            elif observation[0] >= env.goal_position:
                print(f"Completed on episode {episode}")
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

        if (end_epsilon_decaying >= episode >= start_epsilon_decaying):
            epsilon -= epsilon_decay_value

        ep_rewards.append(episode_reward)

        if not episode % show_every:
            average_reward = sum(ep_rewards[-show_every:]) / len(ep_rewards[-show_every:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-show_every:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-show_every:]))

            print(f"Episode: {episode}, average: {average_reward}, min: {min(ep_rewards[-show_every:])}, max: {max(ep_rewards[-show_every:])}")
    env.close()
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
    plt.legend(loc=4)
    plt.show()
