from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, concatenate, Reshape, Conv2D, Bidirectional, GRU
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed


def conv_gru_model():
    # (?, 700)
    main_input = Input(shape=(700,), dtype='int32', name='main_input')
    # (?, 700, 21)
    x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
    # (?, 700, 21)
    auxiliary_input = Input(shape=(700, 21), name='axu_input')
    # (?, 700, 42)
    input_feature = concatenate([x, auxiliary_input], axis=-1)

    # (?, 700, 42, 1)
    conv_input = Reshape((700, 42, 1))(input_feature)
    # (?, 700, 1, 64)
    # Params:conv_a = (42*3 + 1) * 64 = 8128
    conv_a = Conv2D(filters=64, kernel_size=(42, 3), strides=(1, 42),
                    padding='same', activation='relu', kernel_regularizer=l2(0.001))(conv_input)
    # Params:conv_b = (42*7 + 1) * 64 = 18880
    conv_b = Conv2D(filters=64, kernel_size=(42, 7), strides=(1, 42),
                    padding='same', activation='relu', kernel_regularizer=l2(0.001))(conv_input)
    # Params:conv_c = (42*11 + 1) * 64 = 29632
    conv_c = Conv2D(filters=64, kernel_size=(42, 11), strides=(1, 42),
                    padding='same', activation='relu', kernel_regularizer=l2(0.001))(conv_input)
    # (?, 700, 1, 192)
    conv_output = concatenate([conv_a, conv_b, conv_c], axis=-1)
    # (?, 700, 192)
    rnn_input = Reshape((700, 192))(conv_output)

    # (?, 700, 400)
    # Params:gru_1 = bi:2 * gru:3 * ((192+1)*200 + 200*200 + 200*200) = 471600
    gru_1 = Bidirectional(
        GRU(units=200, activation='tanh', recurrent_activation='sigmoid', dropout=0.5, recurrent_dropout=0.5,
            return_sequences=True),
        merge_mode='concat')(rnn_input)
    # Params:gru_2 = bi:2 * gru:3 * ((200+1)*200 + 200*200 + 200*200) = 721200
    gru_2 = Bidirectional(
        GRU(units=200, activation='tanh', recurrent_activation='sigmoid', dropout=0.5, recurrent_dropout=0.5,
            return_sequences=True),
        merge_mode='concat')(gru_1)
    gru_3 = Bidirectional(
        GRU(units=200, activation='tanh', recurrent_activation='sigmoid', dropout=0.5, recurrent_dropout=0.5,
            return_sequences=True),
        merge_mode='concat')(gru_2)
    # (?, 700, 592)
    rnn_output = concatenate([gru_3, rnn_input], axis=-1)

    # (?, 700, 300)
    # Params:fc_1 = (592 + 1) * 300
    fc_1 = TimeDistributed(Dense(300, activation='relu', kernel_regularizer=l2(0.001)))(rnn_output)
    fc_2 = TimeDistributed(Dense(300, activation='relu', kernel_regularizer=l2(0.001)))(fc_1)

    # (?, 700, 8)
    main_output = TimeDistributed(Dense(8, activation='softmax', name='main_output'))(fc_2)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    adam = Adam(lr=0.003)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model

# model = conv_gru_model()
#
# print(model.get_layer(name='main_input'))
