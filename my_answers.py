import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):

    # containers for input/output pairs
    X = []
    y = []

    steps = len(series) - window_size
    from_index = 0
    to_index = window_size

    for _ in range(0,steps):
        X.append(series[from_index:to_index])
        from_index += 1
        to_index += 1
    y = series[window_size:]

     # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    #punctuation = ['!', ',', '.', ':', ';', '?']
    # should I do this with a regEx?
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('-', ' ')
    text = text.replace('*', ' ')
    text = text.replace('/', ' ')
    text = text.replace('&', ' ')
    text = text.replace('%', ' ')
    text = text.replace('@', ' ')
    text = text.replace('$', ' ')
    text = text.replace('à', ' ')
    text = text.replace('â', ' ')
    text = text.replace('è', ' ')
    text = text.replace('é', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    steps = len(text) - window_size
    
    for i in range(0, steps, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
