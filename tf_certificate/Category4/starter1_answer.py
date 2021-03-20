# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000 
    embedding_dim = 16
    max_length = 120 
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000 #20000, 1000, 16, 120

    sentences = []
    labels = []
    # YOUR CODE HERE

    with open ('sarcasm.json', 'r') as f :
        dataset = json.load(f)

    for item in dataset :
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    x_train = sentences[0:training_size]
    x_test = sentences[training_size:]
    y_train = labels[0:training_size]
    y_test = labels[training_size:]

    token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    token.fit_on_texts(x_train)
    token.fit_on_texts(x_test)
    #print(token.word_index)
    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)
    print("max", max(len(I) for I in x_train))
    print("mean", sum(map(len, x_train)) / len(x_train))
    # max 40
    # mean 10.0392

    '''
    vocab_size = 1000 
    embedding_dim = 16
    max_length = 120 
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000 #20000, 1000, 16, 120
    '''
    x_train = pad_sequences(x_train, maxlen=max_length)
    x_test = pad_sequences(x_test, maxlen=max_length)

    #label.nnumpy array
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 120, shuffle = False)

    print(x_train.shape, x_test.shape, x_val.shape) #(16000, 120) (6709, 120) (4000, 120)

    from tensorflow.keras.layers import Conv1D, Flatten, Dense, BatchNormalization
    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),
        # tf.keras.layers.LSTM(64, activation='relu'),
        # tf.keras.layers.Dropout(0.1),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dense(1, activation='sigmoid')
        Conv1D(128, 2, 1, padding = 'same', activation = 'relu'),
        Conv1D(64, 2, 1, padding = 'same', activation = 'relu'),
        Conv1D(32, 2, 1, padding = 'same', activation = 'relu'),
        Flatten(),
        Dense(64, activation = 'relu'),
        BatchNormalization(),
        Dense(32, activation = 'relu'),
        BatchNormalization(),
        Dense(16, activation = 'relu'),
        BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #COMPILE
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.1, verbose=1, mode='auto')
    #path = '../data/modelCheckpoint/category4_tensorflow_embedding_{epoch:02d}-{val_loss:.4f}.hdf5'
    #cp = ModelCheckpoint(path, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
    model.fit(x_train, y_train, epochs = 200, batch_size=128, validation_data= (x_val, y_val), verbose=1, callbacks=[es, lr]) #cp

    #Eval, predict
    loss, acc = model.evaluate(x_test, y_test)
    print("loss : ", loss)
    print("acc : ", acc)

    return model

    # loss :  0.6010524034500122
    # acc :  0.8191980719566345

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
