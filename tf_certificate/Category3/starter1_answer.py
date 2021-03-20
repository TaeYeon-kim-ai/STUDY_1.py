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
# Computer Vision with CNNs
#d
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
    # YOUR CODE HERE
    #DATA
    height_shift_range=(-1,1), 
    width_shift_range=(-1,1), 
    zoom_range=1.1
    )

    train_generator = training_datagen.flow_from_directory(# YOUR CODE HERE
    'C:/STUDY_1.py/tf_certificate/Category3/tmp/rps',
    target_size= (150, 150),
    batch_size= 32
    )

    #print(xy_train) #Found 2520 images belonging to 3 classes.
    
    #MODEL
    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(100,2, padding='SAME',  activation='relu', input_shape = (150, 150, 3)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(128, 2, padding='SAME', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


    #3. COMPILE
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience= 20, verbose= 1, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.3, mode='auto')
    hist = model.fit_generator(train_generator, 
    steps_per_epoch=32, 
    epochs=50, 
    validation_data=train_generator, 
    validation_steps=4,
    callbacks=[es, lr],
    verbose=1
    )

    #4. Eval, Pred
    loss = hist.hist['loss']
    acc = hist.hist['acc']
    print('loss : ', loss)
    print('acc : ', acc)
    
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
