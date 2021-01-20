import numpy as np
import pandas as pd


df = pd.read_csv('./dacon1/data/test/test_merge.csv', index_col=0, header=0, encoding='CP949')
sun_test_data = df.to_numpy()
print(sun_test_data)
print(type(sun_test_data)) # <class 'numpy.ndarray'>
print(sun_test_data.shape) # (2397, 10)
np.save('./dacon1/data/test/sun_test_data.npy', arr=sun_test_data)

