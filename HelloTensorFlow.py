import tensorflow as tf
#import pandas as pd
#import numpy as np
tf.compat.v1.enable_eager_execution()
x = tf.add(1,2)
print(x.numpy())
txt = tf.constant('Hello TensorFlow!')
print(txt.numpy())

