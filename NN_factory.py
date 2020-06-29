from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
import tensorflow as tf

NEURONS = 0
ACTIVATION = 1


# def get_model(params, inputs, outputs, output_activation):
#     input = Input(shape=(inputs,))
#
#     for layer in params:
#         out = Dense(layer[NEURONS], activation=layer[ACTIVATION])


def get_actor(n, bound):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = Input(shape=(n,))
    out = Dense(512, activation="relu")(inputs)
    out = BatchNormalization()(out)
    out = Dense(512, activation="relu")(out)
    out = BatchNormalization()(out)
    outputs = Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * 2
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(n_states, n_actions):
    # State as input
    state_input = Input(shape=n_states)
    state_out = Dense(16, activation="relu")(state_input)
    state_out = BatchNormalization()(state_out)
    state_out = Dense(32, activation="relu")(state_out)
    state_out = BatchNormalization()(state_out)

    action_input = Input(shape=n_actions)
    action_out = Dense(32, activation="relu")(action_input)
    action_out = BatchNormalization()(action_out)

    concat = Concatenate()([state_out, action_out])

    out = Dense(512, activation="relu")(concat)
    out = BatchNormalization()(out)
    out = Dense(512, activation="relu")(out)
    out = BatchNormalization()(out)
    outputs = Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model
