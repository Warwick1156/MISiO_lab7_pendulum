from EnvHelper import EnvHelper
from Buffer import Buffer
from Agent import Agent

# TODO: add parametrization for constructing models
# nn_params = {
#     'actor': ((512, 'relu'), (512, 'relu')),
#     'critic': {
#         'state': ((16, 'relu'), (32, 'relu')),
#         'action': ((32, 'relu')),
#         'connected': ((512, 'relu'), (512, 'relu'))
#     }
# }


if __name__ == '__main__':
    env_helper = EnvHelper()
    env, n_states, n_actions = env_helper.make_environment('BipedalWalker-v3')
    buffer = Buffer(n_states=n_states, n_actions=n_actions, capacity=75000, batch_size=512)
    agent = Agent(gamma=0.99,
                  buffer=buffer,
                  env=env_helper,
                  alpha=0.005,
                  name='WalkerTest',
                  compile_nn=True)
    agent.run(iterations=500, render=False, verbose=True, train=True)
    agent.plot(agent.learning_rewards)
    agent.run(iterations=20, render=True, verbose=True, train=False)
    agent.plot(agent.testing_reward)
