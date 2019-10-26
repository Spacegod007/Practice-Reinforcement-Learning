import gym

from canvas_tutorial.DQN import DQN


def main(envName, trails, trail_len):
    env = gym.make(envName)
    agent = DQN(env=env)

    for trail in range(trails):
        state = env.reset().reshape(1, 2)
        for step in range(trail_len):
            action = agent.act(state)
            env.render()
            observation, reward, done, info = env.step(action)
            # print(reward)
            reward = reward if not done else -20
            observation = observation.reshape(1, 2)
            agent.remember(state, action, reward, observation, done)
            agent.replay()
            agent.target_train()

            if (done):
                break

            state = observation

        if (step >= 199):
            print('Trail {0} failed'.format(trail))
        else:
            print('Completed trail in {0} attempts'.format(trail))
            agent.save_model('successful_model{0}.h5'.format(trail))
            break