import numpy as np
import string

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from typing import List, Tuple


def window_transform_series(
        series: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    '''Transforms the input series and window size into a set of input/output
    pairs for use with our RNN Model

    :param series: numpy array holding the input series
    :param window_size: integer

    :return X, y: tuple of input/output pairs,
    '''
    # Containers for input/output pairs
    X: np.ndarray = []
    y: np.ndarray = []

    # Get sequence length P
    input_size = len(series)

    # Get the number of pairs to create with window size T
    # We create P - T pairs
    num_windows = input_size - window_size + 1

    # Input pair is a vector of length window_size
    # e.g. window size 4 ->
    #   Input: <s1,s2, s3, s4>   Output: s5
    #   Input: <s2, s3, s4, s5>  Output: s6
    for start in range(0, num_windows):
        X.append(series[start: start + window_size])

    # for start in range(0, input_size):
        # end = min(start + window_size, input_size)
        # X.append(series[start:end])

    # Output is the original sequence from window_size
    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


def build_part1_RNN(window_size: int) -> Sequential:
    '''Build an RNN to perform regression on our time series input/output data

    :param window_size: length of window_size to input

    :return sequential model
    '''
    # Init Keras model
    model = Sequential()

    # Create network
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model


def is_ascii_or_punkt(ch: str) -> bool:
    '''Helper function to return whether a character is ascii or in our defined
    punctuation list

    :param ch: char to best test
    :return bool
    '''
    punctuation = ['!', ',', '.', ':', ';', '?']

    if ch == ' ' or ch in string.ascii_lowercase or ch in punctuation:
        return True

    return False


def cleaned_text(text: str) -> str:
    '''Return the text input with only ascii lowercase and punkt
    '''

    # make sure all text is lowercase
    text = text.lower()

    text = ''.join(list(filter(lambda l: is_ascii_or_punkt(l), text)))

    return text


def window_transform_text(text: str,
                          window_size: int,
                          step_size: int) -> Tuple[List[str],
                                                   List[str]]:
    '''Transforms the input text and window size into a set of input/output
    pairs for use with our RNN model

    :param text: our clean corpus
    :param window_size: int, the size of our input (aka window)
    :param step_size: int, how much to slide the window at a time

    :return inputs, outputs: tuple of list of strings,
        inputs -> <s1, s2, s3,...sP> where P is window_size
        outputs -> s6
    '''

    # containers for input/output pairs
    inputs: List = []
    outputs: List = []

    input_size = len(text)

    # Get the number of pairs to create with window size T
    # We create P - T pairs
    num_windows = input_size - window_size + 1

    # Input pair is a vector of length window_size
    # We also take into consideration a step_size
    # e.g. window size 4, step_size 2 ->
    #   Input: <s1,s2, s3, s4>   Output: s5
    # -- Now we shift/step two idxs
    #   Input: <s3,s4, s5, s6>   Output: s7
    for start in range(0, num_windows, step_size):
        inputs.append(text[start: start + window_size])

    # Output starts after window size, every step_size
    for index in range(window_size, input_size, step_size):
        outputs.append(text[index])

    return inputs, outputs


def build_part2_RNN(window_size: int, num_chars: int) -> Sequential:
    '''Build a RNN model with a single LSTM hidden layer with softmax
    activation and categorical cross entropy loss

    :param window_size: int,
    :param num_chars: int,

    :return model: Sequential, keras network
    '''
    # Init Keras Model
    model = Sequential()

    # Create network
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model
