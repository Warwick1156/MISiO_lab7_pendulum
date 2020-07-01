from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
import tensorflow as tf

NEURONS = 0
ACTIVATION = 1


# def get_model(params, inputs, outputs, output_activation):
#     input = Input(shape=(inputs,))
#
#     for layer in params:
#         out = Dense(layer[NEURONS], activation=layer[ACTIVATION])


def get_actor(n, bound, n_actions):
    init_weights = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    input_layer = Input(shape=(n,))
    hidden_layers = Dense(512, activation="relu")(input_layer)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dense(512, activation="relu")(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    output_layer = Dense(n_actions, activation="tanh", kernel_initializer=init_weights)(hidden_layers)
    output_layer = output_layer * bound

    return tf.keras.Model(input_layer, output_layer)


def get_critic(n_states, n_actions):
    nn_1_input_layer = Input(shape=n_states)
    nn_1_hidden = Dense(16, activation="relu")(nn_1_input_layer)
    nn_1_hidden = BatchNormalization()(nn_1_hidden)
    nn_1_hidden = Dense(32, activation="relu")(nn_1_hidden)
    nn_1_hidden = BatchNormalization()(nn_1_hidden)

    nn_2_input_layer = Input(shape=n_actions)
    nn_2_hidden = Dense(32, activation="relu")(nn_2_input_layer)
    nn_2_hidden = BatchNormalization()(nn_2_hidden)
    nn_2_hidden = Dense(32, activation="relu")(nn_2_hidden)
    nn_2_hidden = BatchNormalization()(nn_2_hidden)

    concat = Concatenate()([nn_1_hidden, nn_2_hidden])

    hidden_layers = Dense(512, activation="relu")(concat)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Dense(512, activation="relu")(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    output_layer = Dense(n_actions)(hidden_layers)

    return tf.keras.Model([nn_1_input_layer, nn_2_input_layer], output_layer)
