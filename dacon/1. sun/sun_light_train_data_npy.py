import numpy as np
import pandas as pd

df = pd.read_csv('./dacon1/data/train/train.csv', index_col=0, header=0, encoding='CP949')
sun_train_data = df.to_numpy()
print(sun_train_data)
print(type(sun_train_data)) # <class 'numpy.ndarray'>
print(sun_train_data.shape) # (2397, 10)
np.save('./dacon1/data/test/sun_train_data.npy', arr=sun_train_data)

