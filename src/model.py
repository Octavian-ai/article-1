import tensorflow as tf
import numpy as np


def model_fn(features, labels, mode, params):

	# Retrieve the embedded values for the given node ids
	person_hidden = tf.get_variable("person",   [params["n_person"],  params["embedding_width"]])
	product_hidden = tf.get_variable("product", [params["n_product"], params["embedding_width"]])

	person_emb  = tf.nn.embedding_lookup(person_hidden,  features[0])
	product_emb = tf.nn.embedding_lookup(product_hidden, features[1])

	# Compute the dot-product of the embedded vectors
	m = tf.multiply(person_emb, product_emb)
	m = tf.reduce_sum(m, axis=-1)
	m = tf.expand_dims(m, -1) # So this fits as input for tf.layers api

	# Apply a dense layer and activation function to let the network
	# transform the dot-product to fit the label range
	output = tf.layers.dense(inputs=m, units=(1), activation=tf.nn.sigmoid)

	# Make the size (?, 1) to fit the output of the tf.layers api
	labels = tf.expand_dims(labels, -1)

	# Loss across the batch
	loss = tf.losses.mean_squared_error(output, labels)

	# Let's see the accuracy over time
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(output, labels)
	}

	# tf.summary.scalar("accuracy", eval_metric_ops["accuracy"][1])

	# Train the network to minimise the loss using Adam
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

	# Wrap this up into an EstimatorSpec for Tensorflow to train using high level API
	return tf.estimator.EstimatorSpec(
		mode, 
		loss=loss, 
		train_op=train_op, 
		eval_metric_ops=eval_metric_ops,
		scaffold=scaffold)


