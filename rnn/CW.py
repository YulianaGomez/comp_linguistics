import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras import preprocessing
import string

# Loading
'''
def read(p):
    with open(p) as f:
        lines = f.read()
        lines = lines.split("\n")
    lines = [element.lower() for element in lines]
    return lines

dictionary = read('eng-dictionary.txt')
compound_words = read('english_compound_words.txt')
non_compound_words = open("non_compound", 'a+')

#Creating non_compound file

for i in range(100, 200000):
    if dictionary[i] not in compound_words:
        non_compound_words.write(dictionary[i] + '\n')

'''

# Recurrent Neural Networks from scratch
'''
def rnn(iterations, words_train, label_train ):

    #Input
    inputs = np.tile(np.array(words_train), (iterations, 1))

    #Initial state
    state_t = np.zeros((len(label_train), ))

    #Initial weights
    w = np.random.random((len(label_train), len(words_train)))
    u = np.random.random((len(label_train), len(words_train)))
    b = np.random.random((len(label_train), ))

    successive_outputs = []
    for input_t in inputs:
        output_t = np.tanh(np.dot(w, input_t) + np.dot(u, state_t) + b)
        state_t = output_t

    return state_t
'''

# Creating dataframe

df1 = pd.read_csv('english_compound_words1.txt', sep="\t", header=None, names=["Word", "Label"])
df1.Word = df1.Word.str.lower()
df2 = pd.read_csv('non_compound_english1.txt', sep="\t", header=None, names=["Word", "Label"])
df = pd.concat([df1,df2])
df = df.sample(n = df.shape[0], random_state = 1)


# Training and Test sets

words_train, words_test, label_train, label_test = train_test_split(df.Word, df.Label, test_size=0.3)

#Text-vectorization

def one_hot(words_train):

    characters = string.printable
    token_index = dict(zip(characters, range(1, len(characters) + 1)))

    max_length = 10000
    results = np.zeros((len(words_train), max_length, max(token_index.keys())+1))
    for i, words_train in enumerate(words_train):
        for j, character in enumerate(words_train):
            index = token_index.get(character)
            results[i,j,index] = 1

    return results


# Keras version with embedding layer

'''
model = Sequential()
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fix(words_train, label_train, epochs = 10, batch_size = 100, validation_split = 0.2)
'''

# Test

one_hot(words_train)

