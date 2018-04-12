import tensorflow as tf
import numpy as np

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



	# --------------------------------------------------------------------------
	# Model
	# --------------------------------------------------------------------------
	
	person_hidden  = tf.get_variable("person",  [params["n_person"],  params["embedding_width"]])
	product_hidden = tf.get_variable("product", [params["n_product"], params["embedding_width"]])

	person_emb  = tf.nn.embedding_lookup(person_hidden,  person_id)
	product_emb = tf.nn.embedding_lookup(product_hidden, product_id)

	# Compute the dot-product of the embedded vectors
	m = tf.multiply(person_emb, product_emb)
	m = tf.reduce_sum(m, axis=-1)
	m = tf.expand_dims(m, -1) # So this fits as input for tf.layers api

	# Apply a dense layer and activation function to let the network
	# transform the dot-product to fit the label range
	pred_review_score = tf.layers.dense(inputs=m, units=(1), activation=tf.nn.sigmoid)



	# --------------------------------------------------------------------------
	# Build EstimatorSpec
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

		label_review_score 	= labels

		# Make the size (?, 1) to fit the pred_review_score of the tf.layers api
		label_review_score = tf.expand_dims(label_review_score, -1)

		# Loss across the batch
		loss = tf.losses.mean_squared_error(pred_review_score, label_review_score)

		classes = 2

		# Let's see the accuracy over time
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(pred_review_score, label_review_score),
			"accuracy_per_class": tf.metrics.mean_per_class_accuracy(
				score_to_class(label_review_score, classes),
				score_to_class(pred_review_score, classes), classes)
		}

		train_op = tf.train.AdamOptimizer(params["lr"]).minimize(loss=loss, global_step=tf.train.get_global_step())

		# Initialise the embedding without upsetting graph size limits (https://stackoverflow.com/questions/48217599/how-to-initialize-embeddings-layer-within-estimator-api/48243086#48243086)
		def init_fn(scaffold, sess):
			try:
				person_initial = np.random.uniform(-1.0, 1.0, person_hidden.shape.as_list()).astype(np.float32)
				sess.run(person_hidden.initializer, {person_hidden.initial_value: person_initial})

				product_initial = np.random.uniform(-1.0, 1.0, product_hidden.shape.as_list()).astype(np.float32)
				sess.run(product_hidden.initializer, {product_hidden.initial_value: product_initial})
			except:
				traceback.print_exc()
		
		scaffold = tf.train.Scaffold(init_fn=init_fn)

		return tf.estimator.EstimatorSpec(
			mode, 
			loss=loss, 
			train_op=train_op, 
			eval_metric_ops=eval_metric_ops,
			scaffold=scaffold)

	# --------------------------------------------------------------------------


	if mode == tf.estimator.ModeKeys.PREDICT:

		label_review_score = features[4]
		label_review_score = tf.expand_dims(label_review_score, -1)
		loss = tf.square(tf.abs(label_review_score - pred_review_score))

		predictions = {
			"person_id": person_id,
			"product_id": product_id,
			"pred_review_score": tf.squeeze(pred_review_score, -1),
			"label_review_score": tf.squeeze(label_review_score, -1),
			"person_emb": person_emb,
			"product_emb": product_emb,
			"product_style": product_style,
			"person_style": person_style,
			"loss": tf.squeeze(loss, -1),
			"label_review_score_check": tf.reduce_sum(tf.multiply(product_style, person_style), axis=-1),
		}

		return tf.estimator.EstimatorSpec(
			mode, 
			predictions=predictions)

	
	# --------------------------------------------------------------------------



	

	


