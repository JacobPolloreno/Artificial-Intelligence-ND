from keras import backend as K
from keras.models import Model
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Input,
    TimeDistributed,
    Activation,
    Bidirectional,
    SimpleRNN,
    GRU,
    LSTM,
    MaxPooling1D)


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                   implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(
        units,
        activation=activation,
        return_sequences=True,
        implementation=2,
        name='rnn')(input_data)
    # Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(
        units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        name='rnn')(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name="bn_rnn_1")(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def cnn_pooling_length(cnn_output_length, pool_ksize, pool_border_mode,
                       pool_stride):

    assert pool_border_mode == 'valid'
    return (cnn_output_length - pool_ksize) // pool_stride + 1


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layers, each with batch normalization
    for l in range(1, recur_layers + 1):
        if l == 1:
            rnn = GRU(
                units,
                activation='relu',
                return_sequences=True,
                implementation=2,
                name="rnn_{}".format(l))(input_data)
        else:
            rnn = GRU(
                units,
                activation='relu',
                return_sequences=True,
                implementation=2,
                name="rnn_{}".format(l))(bn_rnn)
        bn_rnn = BatchNormalization(name="bn_rnn_{}".format(l))(rnn)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(
        units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        name="bidir_rnn"),
        merge_mode='concat')(input_data)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def dilation_cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                           conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    that takes advantage of dilated convolutions

    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))


def final_model(input_dim, filters, kernel_size, conv_stride,
                conv_border_mode, pool_ksize, pool_stride,
                pool_border_mode, rnn_units, dropout_prob,
                cell_type: str = 'GRU', output_dim=29):
    """ Build a deep network for speech

    CNN -> BN -> MaxPooling1D -> 3x Bidirectional RNNs
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add CNN layer
    conv_1d_1 = Conv1D(filters, kernel_size,
                       strides=conv_stride,
                       padding=conv_border_mode,
                       activation='relu',
                       name='conv1d_1')(input_data)
    bn_cnn_1 = BatchNormalization(name='bn_conv1d_1')(conv_1d_1)
    activation_conv_1 = Activation('relu', name="activation_conv_1")(bn_cnn_1)
    maxpool_1d_1 = MaxPooling1D(pool_size=pool_ksize,
                                strides=pool_stride,
                                padding=pool_border_mode,
                                name="maxpool_1d_1")(activation_conv_1)

    if cell_type.lower() == 'gru':
        cell = GRU
    elif cell_type.lower() == 'lstm':
        cell = LSTM
    elif cell_type.lower() == 'simple':
        cell = SimpleRNN
    else:
        raise ValueError(
            "Invalid Recurrent Cell Type. Valid choices: {GRU, LSTM, SIMPLE")

    # Add BRNN stacks
    bidir_rnn_1 = Bidirectional(cell(
        rnn_units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        dropout=dropout_prob),
        name="brnn_{}_1".format(cell_type.lower()))(maxpool_1d_1)

    bidir_rnn_2 = Bidirectional(cell(
        rnn_units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        dropout=dropout_prob),
        name="brnn_{}_2".format(cell_type.lower()))(bidir_rnn_1)

    bidir_rnn_3 = Bidirectional(cell(
        rnn_units,
        activation='relu',
        return_sequences=True,
        implementation=2,
        dropout=dropout_prob),
        name="brnn_{}_3".format(cell_type.lower()),
        merge_mode='concat')(bidir_rnn_2)

    bn_1 = BatchNormalization(name='bn_1')(bidir_rnn_3)

    # Add a TimeDistributed layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_1)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_pooling_length(
        cnn_output_length(x, kernel_size, conv_border_mode, conv_stride),
        pool_ksize, pool_border_mode, pool_stride)
    return model
