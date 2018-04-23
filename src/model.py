import tensorflow as tf
import numpy as np

import traceback
from functools import reduce

def score_to_class(tensor, buckets=2):
	return tf.cast(tf.round(tensor * (buckets-1)), tf.int32)

def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Model
	# --------------------------------------------------------------------------

	nouns = ["person", "product"]
	hidden = {}
	embeddings = {}

	for noun in nouns:
		hidden[noun] = tf.get_variable(noun, [params["n_"+noun],  params["embedding_width"]])
		emb  = tf.nn.embedding_lookup(hidden[noun], features[noun]["id"])
		embeddings[noun] = emb

	# Compute the dot-product of the embedded vectors
	m = tf.multiply(*embeddings.values())
	m = tf.reduce_sum(m, axis=-1)
	m = tf.expand_dims(m, -1) # So this fits as input for tf.layers api

	# Apply a dense layer and activation function to let the network
	# transform the dot-product to fit the label range
	pred_review_score = tf.layers.dense(inputs=m, units=(1), activation=tf.nn.sigmoid)


	# --------------------------------------------------------------------------
	# Build EstimatorSpec
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

		label_review_score = labels

		# Make the size (?, 1) to fit the pred_review_score of the tf.layers api
		label_review_score = tf.expand_dims(label_review_score, -1)
		loss = tf.losses.mean_squared_error(pred_review_score, label_review_score)

		classes = 2

		# Let's see the accuracy over time
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(pred_review_score, label_review_score),
			"accuracy_per_class": tf.metrics.mean_per_class_accuracy(
				score_to_class(label_review_score, classes),
				score_to_class(pred_review_score, classes), classes),
		}

		train_op = tf.train.AdamOptimizer(params["lr"]).minimize(loss, global_step=tf.train.get_global_step())

		return tf.estimator.EstimatorSpec(
			mode, 
			loss=loss, 
			train_op=train_op, 
			eval_metric_ops=eval_metric_ops
		)

	# --------------------------------------------------------------------------


	if mode == tf.estimator.ModeKeys.PREDICT:

		predictions = {
			"review_score": tf.squeeze(pred_review_score, -1),
		}

		return tf.estimator.EstimatorSpec(
			mode, 
			predictions=predictions)

	
	# --------------------------------------------------------------------------



	

	


