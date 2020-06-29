import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from Buffer import Batch
from NN_factory import get_actor, get_critic
from OUP import Noise


class Agent:
    def __init__(self, env, gamma, buffer, alpha, name='Default_Agent_Name'):
        self.buffer = buffer
        self.batch = Batch()
        self.env_helper = env
        self.env = self.env_helper.env

        self.critic = get_critic(self.env_helper.n_states, self.env_helper.n_actions)
        self.critic_target = get_critic(self.env_helper.n_states, self.env_helper.n_actions)

        self.actor = get_actor(self.env_helper.n_states, self.env_helper.action_bounds[0][1])
        self.actor_target = get_actor(self.env_helper.n_states, self.env_helper.action_bounds[0][1])
        # self.actor = get_actor(self.env_helper.n_states, 2)
        # self.actor_target = get_actor(self.env_helper.n_states, 2)

        self.gamma = gamma
        self.alpha = alpha
        self.name = name

        self.critic_optimizer = tf.optimizers.Adam(0.002)
        self.actor_optimizer = tf.optimizers.Adam(0.001)

        self.exploration_policy = Noise(mean=np.zeros(1), std=0.2 * np.ones(1))
        self.learning_rewards = []
        self.testing_reward = []

    def learn(self):
        self.batch.get_batch(self.buffer)

        with tf.GradientTape() as tape:
            target_actions = self.actor_target(self.batch.next_state)
            expected_return = self.batch.reward + self.gamma * self.critic_target([self.batch.next_state, target_actions])
            critic_value = self.critic([self.batch.state, self.batch.action])
            critic_loss = tf.math.reduce_mean(tf.math.square(expected_return - critic_value))

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer .apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(self.batch.state)
            critic_value = self.critic([self.batch.state, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

    def update_target(self):
        new_weights = []
        target_variables = self.critic_target.weights
        for i, variable in enumerate(self.critic.weights):
            new_weights.append(variable * self.alpha + target_variables[i] * (1 - self.alpha))
        self.critic_target.set_weights(new_weights)

        new_weights = []
        target_variables = self.actor_target.weights
        for i, variable in enumerate(self.actor.weights):
            new_weights.append(variable * self.alpha + target_variables[i] * (1 - self.alpha))
        self.actor_target.set_weights(new_weights)

    def policy(self, state, train):
        prediction = tf.squeeze(self.actor(state))
        noise = 0
        if train:
            noise = self.exploration_policy()
        # noise = self.exploration_policy()
        noisy_prediction = prediction.numpy() + noise

        # TODO: generalize to any number of actions
        legal_action = np.clip(noisy_prediction, self.env_helper.action_bounds[0][0], self.env_helper.action_bounds[0][1])
        # legal_action = np.clip(sample, -2, 2)

        return [np.squeeze(legal_action)]

    def run(self, iterations, render, verbose, train=True):
        if not train:
            self.load()
        for i in range(iterations):
            previous_state = self.env.reset()
            iteration_reward = 0

            while True:
                if render:
                    self.env.render()
                previous_state_tensor = tf.expand_dims(tf.convert_to_tensor(previous_state), 0)
                action = self.policy(previous_state_tensor, train)
                state, reward, done, info = self.env.step(action)

                if train:
                    self.buffer.record((previous_state, action, reward, state))
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

            if verbose:
                print('Iteration: {0}, score: {1}'.format(i, iteration_reward))
        self.save()

    def plot(self, data):
        plt.plot(data)
        plt.xlabel("Iteracja")
        plt.ylabel("Nagroda")
        plt.show()

    def save(self):
        self.actor.save(self.name + "_actor")
        self.critic.save(self.name + "_critic")
        self.actor_target.save(self.name + "_actor_target")
        self.critic_target.save(self.name + "_critic_target")

    def load(self):
        self.critic = load_model(self.name + "_actor")
        self.critic_target = load_model(self.name + "_critic")

        self.actor = load_model(self.name + "_actor_target")
        self.actor_target = load_model(self.name + "_critic_target")
