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
# QUESTION
#
# Build and train a neural network to predict sunspot activity using
# the Sunspots.csv dataset.
#
# Your neural network must have an MAE of 0.12 or less on the normalized dataset
# for top marks.
#
# Code for normalizing the data is provided and should not be changed.
#
# At the bottom of this file, we provide  some testing
# code in case you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.


import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1) #.Series(['Kim', 'Lee', 'Park'])
    ds = tf.data.Dataset.from_tensor_slices(series) #
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True) #window_size = 그룹화할 윈도우 크기 / drop_remainder 남은 부분을 버릴지 살릴지/ shift = 1iteration당 몇개씩 이동할건지
    ds = ds.flat_map(lambda w: w.batch(window_size + 1)) # 데이터셋에 함수 apply해주고 결과 flatten하게 펼쳐줌
    ds = ds.shuffle(shuffle_buffer) # 데이터셋 섞어줌
    ds = ds.map(lambda w: (w[:-1], w[1:])) # map(변환 함수, 순회 가능한 데이터) // 
    return ds.batch(batch_size).prefetch(1) #



def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')
    
    
    time_step = []
    sunspots = []
    
    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))#sunspots의 0번째 열을 가져와서 sunspots에 넣어줌 # YOUR CODE HERE) 
        time_step.append(int(row[0]))#sunspots의 1번째 열을 가져와서 time_strp에 넣어줌 #  YOUR CODE HERE)

    series = np.array(sunspots)# 열번호를 YOUR CODE HERE 
    time = np.array(time_step) 
    # print(time_step)
    # print(sunspots)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000
    time_train = time[:split_time] #  time 3000개까지 가져옴# YOUR CODE HERE
    x_train = series[:split_time]# # series 3000개 가져옴 #YOUR CODE HERE
    time_valid = time[split_time:]# time 3000개 이후 남은것 가져옴 # YOUR CODE HERE
    x_valid = series[split_time:]# series 3000개 이후 남은것 가져옴#YOUR CODE HERE
    print(time_train.shape) #(3000,)
    print(x_train.shape) #(3000,)
    print(time_valid.shape) #(235,)
    print(x_valid.shape) #(235,)
    
    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
    print(train_set)
    print(x_train.shape)
    
    model = tf.keras.models.Sequential([
      # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
      tf.keras.layers.Conv1D(32, 4, strides=1, activation='relu', input_shape = (None, 1)),
      tf.keras.layers.LSTM(32, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'), 
      tf.keras.layers.Dense(1)
    ])
    model.summary()
    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    # 	 [[{{node IteratorGetNext}}]]
    #
    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    model.compile(loss = 'mae', optimizer='adam', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto')
    model.fit(train_set, epochs = 200, verbose = 1, callbacks=[es, lr])
  # THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
  # BEFORE UPLOADING YOU CAN DO IT WITH THIS
    def model_forecast(model, series, window_size):
      ds = tf.data.Dataset.from_tensor_slices(series)
      ds = ds.window(window_size, shift=1, drop_remainder=True)
      ds = ds.flat_map(lambda w: w.batch(window_size))
      ds = ds.batch(32).prefetch(1)
      forecast = model.predict(ds)
      return forecast
    window_size = 30# YOUR CODE HERE
    rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

    result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

    # To get the maximum score, your model must have an MAE OF .12 or less.
    # When you Submit and Test your model, the grading infrastructure
    # converts the MAE of your model to a score from 0 to 5 as follows:

    test_val = 100 * result
    score = math.ceil(17 - test_val)
    if score > 5:
      score = 5

    print(score)

    return model

    
# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")



# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS



