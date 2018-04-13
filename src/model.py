import tensorflow as tf
import numpy as np

import traceback
from functools import reduce

def score_to_class(tensor, buckets=2):
	return tf.cast(tf.round(tensor * (buckets-1)), tf.int32)

def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Inputs
	# --------------------------------------------------------------------------

	person_id 			= features[0]
	product_id 			= features[1]
	person_style 		= features[2] # For prediction debugging
	product_style 		= features[3]

	features_d = {
		"person": {
			"id": features[0],
			"style": features[2]
		},
		"product": {
			"id": features[1],
			"style": features[3]
		}
	}



	# --------------------------------------------------------------------------
	# Model
	# --------------------------------------------------------------------------

	nouns = ["person", "product"]
	embeddings = []
	cluster_losses = []

	for noun in nouns:
		hidden = tf.get_variable(noun, [params["n_"+noun],  params["embedding_width"]])
		emb  = tf.nn.embedding_lookup(hidden, features_d[noun]["id"])
		# emb = tf.layers.dense(inputs=emb, units=(params["embedding_width"]), activation=tf.nn.relu)
		embeddings.append(emb)

		# K-Means
		clusters = tf.get_variable(noun+"_cluster", [params["n_cluster"],  params["embedding_width"]])
		cluster_dist = tf.square(tf.expand_dims(clusters, 0) - tf.expand_dims(emb, 1))
		closest = tf.reduce_min(cluster_dist)
		cluster_losses.append(closest)



	# Compute the dot-product of the embedded vectors
	m = tf.multiply(*embeddings)
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
		emb_loss = tf.losses.mean_squared_error(pred_review_score, label_review_score)

		cluster_loss = tf.reduce_sum(reduce((lambda a,b: a+b), cluster_losses ))

		loss = tf.convert_to_tensor(params["cluster_factor"]) * cluster_loss + emb_loss

		classes = 2

		# Let's see the accuracy over time
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(pred_review_score, label_review_score),
			"accuracy_per_class": tf.metrics.mean_per_class_accuracy(
				score_to_class(label_review_score, classes),
				score_to_class(pred_review_score, classes), classes)
		}

		train_op = tf.train.AdamOptimizer(params["lr"]).minimize(loss=loss, global_step=tf.train.get_global_step())

		return tf.estimator.EstimatorSpec(
			mode, 
			loss=loss, 
			train_op=train_op, 
			eval_metric_ops=eval_metric_ops
		)

	# --------------------------------------------------------------------------


	if mode == tf.estimator.ModeKeys.PREDICT:

		# label_review_score = features[4]
		# label_review_score = tf.expand_dims(label_review_score, -1)
		# loss = tf.square(tf.abs(label_review_score - pred_review_score))

		predictions = {
			"person_id": person_id,
			"product_id": product_id,
			"pred_review_score": tf.squeeze(pred_review_score, -1),
			"label_review_score": tf.squeeze(label_review_score, -1),
			"person_emb": person_emb,
			"product_emb": product_emb,
			"product_style": product_style,
			"person_style": person_style,
			# "loss": tf.squeeze(loss, -1),
			"label_review_score_check": tf.reduce_sum(tf.multiply(product_style, person_style), axis=-1),
		}

		return tf.estimator.EstimatorSpec(
			mode, 
			predictions=predictions)

	
	# --------------------------------------------------------------------------



	

	


