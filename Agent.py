import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from Buffer import Batch
from NN_factory import get_actor, get_critic
from OUP import Noise

ACTOR = '_ACTOR'
CRITIC = '_CRITIC'
TARGET = '_TARGET'


class Agent:
    def __init__(self, env, gamma, buffer, alpha, name='Default_Agent_Name', compile_nn=True):
        self.buffer = buffer
        self.batch = Batch()
        self.env_helper = env
        self.env = self.env_helper.env

        self.gamma = gamma
        self.alpha = alpha
        self.name = name

        self.learning_rewards = []
        self.testing_reward = []

        self.critic = None
        self.critic_target = None
        self.actor = None
        self.actor_target = None
        # self.actor = get_actor(self.env_helper.n_states, 2)
        # self.actor_target = get_actor(self.env_helper.n_states, 2)
        self.critic_optimizer = None
        self.actor_optimizer = None
        self.exploration_policy = None

        if compile_nn:
            self.set_up_learning_components()

    def learn(self):
        self.batch.get_batch(self.buffer)
        self.learn_critic()
        self.learn_actor()

    def update_target(self):
        self.update_critic_target_weights()
        self.update_actor_target_weights()

    def action(self, state, train):
        prediction = tf.squeeze(self.actor(state))
        noise = 0
        if train:
            noise = self.exploration_policy()
        noisy_prediction = prediction.numpy() + noise

        # TODO: generalize to any number of actions
        # As I see in environment implementation, I don't have to clip prediction here
        #legal_action = np.clip(noisy_prediction, self.env_helper.action_bounds[0][0], self.env_helper.action_bounds[0][1])
        # legal_action = np.clip(sample, -2, 2)

        # return [np.squeeze(legal_action)]
        # return [np.squeeze(noisy_prediction)]
        return noisy_prediction

    def run(self, iterations=100, render=False, verbose=False, train=True):
        self.set_up_neural_networks(train)

        for i in range(iterations):
            reward = self.run_iteration(train, render)
            if verbose:
                print('Iteration: {0}, score: {1}'.format(i, reward))

        if train:
            self.save()

    @staticmethod
    def plot(data):
        plt.plot(data)
        plt.xlabel("Iteracja")
        plt.ylabel("Nagroda")
        plt.show()

    def save(self):
        self.actor.save(self.name + ACTOR)
        self.critic.save(self.name + CRITIC)
        self.actor_target.save(self.name + ACTOR + TARGET)
        self.critic_target.save(self.name + CRITIC + TARGET)

    def load(self):
        self.critic = load_model(self.name + ACTOR)
        self.critic_target = load_model(self.name + CRITIC)

        self.actor = load_model(self.name + ACTOR + TARGET)
        self.actor_target = load_model(self.name + ACTOR + TARGET)

    def set_up_learning_components(self):
        self.critic = get_critic(self.env_helper.n_states, self.env_helper.n_actions)
        self.critic_target = get_critic(self.env_helper.n_states, self.env_helper.n_actions)

        self.actor = get_actor(self.env_helper.n_states, self.env_helper.action_bounds[0][1], self.env_helper.n_actions)
        self.actor_target = get_actor(self.env_helper.n_states, self.env_helper.action_bounds[0][1], self.env_helper.n_actions)
        # self.actor = get_actor(self.env_helper.n_states, 2)
        # self.actor_target = get_actor(self.env_helper.n_states, 2)

        self.critic_optimizer = tf.optimizers.Adam(0.002)
        self.actor_optimizer = tf.optimizers.Adam(0.001)

        self.exploration_policy = Noise(mean=np.zeros(1), std=0.2 * np.ones(1))

    def update_critic_target_weights(self):
        new_weights = []
        target_variables = self.critic_target.weights
        for i, variable in enumerate(self.critic.weights):
            new_weights.append(variable * self.alpha + target_variables[i] * (1 - self.alpha))
        self.critic_target.set_weights(new_weights)

    def update_actor_target_weights(self):
        new_weights = []
        target_variables = self.actor_target.weights
        for i, variable in enumerate(self.actor.weights):
            new_weights.append(variable * self.alpha + target_variables[i] * (1 - self.alpha))
        self.actor_target.set_weights(new_weights)

    def learn_actor(self):
        with tf.GradientTape() as tape:
            actions = self.actor(self.batch.state)
            critic_value = self.critic([self.batch.state, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

    def learn_critic(self):
        with tf.GradientTape() as tape:
            target_actions = self.actor_target(self.batch.next_state)
            expected_return = self.batch.reward + self.gamma * self.critic_target(
                [self.batch.next_state, target_actions])
            critic_value = self.critic([self.batch.state, self.batch.action])
            critic_loss = tf.math.reduce_mean(tf.math.square(expected_return - critic_value))
        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

    def run_iteration(self, train, render):
        previous_state = self.env.reset()
        iteration_reward = 0
        while True:
            if render:
                self.env.render()
            previous_state_tensor = tf.expand_dims(tf.convert_to_tensor(previous_state), 0)
            action = self.action(previous_state_tensor, train)
            state, reward, done, info = self.env.step(action)

            if train:
                self.buffer.record(previous_state, action, reward, state)
                self.learn()
                self.update_target()

            iteration_reward += reward
            if done:
                break
            previous_state = state

        if train:
            self.learning_rewards.append(iteration_reward)
        else:
            self.testing_reward.append(iteration_reward)

        return iteration_reward

    def set_up_neural_networks(self, train):
        if not train:
            self.load()
        else:
            if self.actor is None:
                self.set_up_learning_components()
