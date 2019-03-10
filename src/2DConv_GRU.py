import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, concatenate, Reshape, Conv2D, Bidirectional, GRU
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics


def load_cul6133():
    '''
    TRAIN data Cullpdb+profile_6133_filtered
    Test data  CB513\CASP10\CASP11
    '''
    print("Loading train data (Cullpdb_filted)...")
    # data = np.load(gzip.open('../cullpdb_5926_data/cullpdb+profile_5926.npy.gz', 'rb'))
    data = np.load(open('cullpdb+profile_6133.npy', 'rb'))
    data = np.reshape(data, (-1, 700, 57))

    datahot = data[:, :, 0:21]  # sequence feature
    # print 'sequence feature',dataonehot[1,:3,:]
    datapssm = data[:, :, 35:56]  # profile feature
    # print 'profile feature',datapssm[1,:3,:]
    labels = data[:, :, 22:30]  # secondary struture label , 8-d
    # shuffle data
    num_seqs, seqlen, feature_dim = np.shape(data)
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)  #
    np.random.shuffle(seq_index)

    # train data
    trainhot = datahot[seq_index[:5600]]  # 21
    trainlabel = labels[seq_index[:5600]]  # 8
    trainpssm = datapssm[seq_index[:5600]]  # 21

    # val data
    vallabel = labels[seq_index[5605:5877]]  # 8
    valpssm = datapssm[seq_index[5605:5877]]  # 21
    valhot = datahot[seq_index[5605:5877]]  # 21

    # test data
    testhot = datahot[seq_index[5877:]]  # 21
    testlabel = labels[seq_index[5877:]]  # 8
    testpssm = datapssm[seq_index[5877:]]  # 21

    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    for i in range(trainhot.shape[0]):
        for j in range(trainhot.shape[1]):
            if np.sum(trainhot[i, j, :]) != 0:
                train_hot[i, j] = np.argmax(trainhot[i, j, :])

    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    for i in range(valhot.shape[0]):
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i, j, :]) != 0:
                val_hot[i, j] = np.argmax(valhot[i, j, :])

    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i, j, :]) != 0:
                val_hot[i, j] = np.argmax(testhot[i, j, :])

    return train_hot, trainpssm, trainlabel, val_hot, valpssm, vallabel, test_hot, testlabel, testpssm


def conv_gru_model():
    # (?, 700)
    main_input = Input(shape=(700,), dtype='int32', name='main_input')
    # (?, 700, 21)
    x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
    # (?, 700, 21)
    auxiliary_input = Input(shape=(700, 21), name='aux_input')
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
    fc_1 = TimeDistributed(
        Dense(300, activation='relu', kernel_regularizer=l2(0.001)),
    )(rnn_output)
    fc_2 = TimeDistributed(
        Dense(300, activation='relu', kernel_regularizer=l2(0.001)),
    )(fc_1)

    # (?, 700, 8)
    main_output = TimeDistributed(
        Dense(8, activation='softmax'),
        name='main_output'
    )(fc_2)

    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])
    adam = Adam(lr=0.003)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


trainhot, trainpssm, trainlabel, valhot, valpssm, vallabel, testhot, testlabel, testpssm = load_cul6133()

print(trainhot.shape)
print(trainpssm.shape)
print(trainlabel.shape)
print(valhot.shape)
print(valpssm.shape)
print(vallabel.shape)
print(testhot.shape)
print(testlabel.shape)
print(testpssm.shape)

conv_gru_model = conv_gru_model()

# earlyStopping = EarlyStopping(monitor='val_weighted_accuracy', patience=5, verbose=1, mode='auto')
load_file = 'a3-7-11_64conv2d-3_200bi_gru-2_300fc-adam003.h5'
checkpointer = ModelCheckpoint(filepath=load_file, verbose=1, save_best_only=True)

history = conv_gru_model.fit({'main_input': trainhot, 'aux_input': trainpssm},
                             {'main_output': trainlabel},
                             validation_data=({'main_input': valhot, 'aux_input': valpssm}, {'main_output': vallabel}),
                             epochs=200,
                             batch_size=64,
                             # callbacks=[checkpointer, earlyStopping],
                             callbacks=[checkpointer],
                             verbose=2,
                             shuffle=True,
                             )

# model.load_weights(load_file)
