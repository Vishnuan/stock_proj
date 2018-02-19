#!/home/thefishvish/anaconda3/envs/test/lib/python3.6

"""Simple Tensorflow Esimator used for reference

"""

import tensorflow as tf 
import numpy as np

print("Using TensorFlow version "+tf.__version__)

from tensorflow.contrib.learn.python.learn.datasets import base

import iris_data

(train_x, train_y), (test_x, test_y) = iris_data.load_data()

# print(train_x.shape, train_y.shape) #ensure that the data loaded sucessfully using a prebuilt library

#---- build model here---

feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name,shape=[4])]

classifier = tf.estimator.LinearClassifier(
	feature_columns=feature_columns,
	n_classes=3,
	model_dir="tmp/iris_model")
#--model done
#--input function--

def input_fn(x_tr,y_tr):
	def _fn():
		features = {feature_name: tf.constant(x_tr)}
		label = tf.constant(y_tr)
		return features,label
	return _fn

# print(input_fn(train_x, train_y)()) # check to see if our data is being converted to a tensor

#--time for classifier--

classifier.train(input_fn=input_fn(train_x, train_y),steps=1000)

print("fit Done")

accu = classifier.evaluate(input_fn=input_fn(test_x, test_y),steps=100)["accuracy"]

print("\nAccuracy : {0:f}".format(accu))
