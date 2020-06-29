import tensorflow as tf
import numpy as np


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3


class Buffer:
    def __init__(self, n_states, n_actions, capacity=100000, batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.counter = 0

        self.state = np.zeros((self.capacity, n_states))
        self.action = np.zeros((self.capacity, n_actions))
        self.reward = np.zeros((self.capacity, 1))
        self.next_state = np.zeros((self.capacity, n_states))

    def record(self, observation):
        index = self.counter % self.capacity
        # print(index)

        self.state[index] = observation[STATE]
        self.action[index] = observation[ACTION]
        self.reward[index] = observation[REWARD]
        self.next_state[index] = observation[NEXT_STATE]

        self.counter += 1
        # print(self.counter)


class Batch:
    def __init__(self):
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None

    def get_batch(self, buffer):
        record_range = min(buffer.counter, buffer.capacity)
        batch_indices = np.random.choice(record_range, buffer.batch_size)

        self.state = tf.convert_to_tensor(buffer.state[batch_indices])
        self.action = tf.convert_to_tensor(buffer.action[batch_indices])
        self.reward = tf.convert_to_tensor(buffer.reward[batch_indices])
        self.reward = tf.cast(self.reward, dtype=tf.float32)
        self.next_state = tf.convert_to_tensor(buffer.next_state[batch_indices])

