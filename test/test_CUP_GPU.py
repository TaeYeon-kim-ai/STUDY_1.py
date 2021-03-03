
import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from IPython.display import display
import PIL


# How to check if the code is running on GPU or CPU?

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

[name: "/cpu:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 4800764240957379342
, name: "/gpu:0"
device_type: "GPU"
memory_limit: 6814913823
locality {
  bus_id: 1
}
incarnation: 14858485129082007400
physical_device_desc: "device: 0, name: GeForce RTX 3090, pci bus id: 0000:01:00.0"
]

# How to check if Keras is using GPU?

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

['/gpu:0']




