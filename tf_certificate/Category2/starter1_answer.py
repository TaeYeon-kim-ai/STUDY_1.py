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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    #1. DATA
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #(60000, 28, 28) (10000, 28, 28) (60000,) (10000,)1
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size = 0.9, random_state = 128, shuffle = True)
    #print(x_train.shape, x_test.shape, x_val.shape) 
    #(54000, 28, 28) (10000, 28, 28) (6000, 28, 28)

    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    #2. MODEL
    inputs = Input(shape = (28, 28))
    x = Conv1D(64, kernel_size= 2, padding='SAME', activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 2, padding='SAME', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)

    x = Conv1D(32, 2, padding='SAME', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()

    #3. COMPILE
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    es = EarlyStopping(monitor='val_loss', patience= 20, verbose=1, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor= 0.3, verbose=1, mode='auto')
    model.fit(x_train, y_train, epochs = 200, batch_size=32, validation_data = (x_val, y_val), verbose=1 , callbacks= [es, lr])

    #4. Eval, Pred
    loss, acc = model.evaluate(x_test, y_test)
    print("loss : ", loss)
    print("acc  : ", acc)

    # loss :  0.5194945335388184
    # acc  :  0.90420001745224
    
    # y_pred = model.predict(x_test)
    # print(y_pred[10])
    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
