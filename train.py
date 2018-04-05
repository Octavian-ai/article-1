
import traceback
import argparse

import tensorflow as tf
import numpy as np

from data import GraphData


def train(args):

	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	def model_fn(features, labels, mode):

		# Retrieve the embedded values for the given node ids
		person_hidden = tf.get_variable("person", [len(person_ids), args.embedding_width])
		product_hidden = tf.get_variable("product", [len(product_ids), args.embedding_width])

		person_emb  = tf.nn.embedding_lookup(person_hidden , features[0])
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
		train_op = tf.train.AdamOptimizer(args.lr).minimize(loss=loss, global_step=tf.train.get_global_step())

		# Initialise the embedding without upsetting graph size limits (https://stackoverflow.com/questions/48217599/how-to-initialize-embeddings-layer-within-estimator-api/48243086#48243086)
		def init_fn(scaffold, sess):
			try:
				person_initial = np.random.uniform(-1.0, 1.0, [len(person_ids), args.embedding_width]).astype(np.float32)
				sess.run(person_hidden.initializer, {person_hidden.initial_value: person_initial})

				product_initial = np.random.uniform(-1.0, 1.0, [len(product_ids), args.embedding_width]).astype(np.float32)
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

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=args.model_dir)

	max_steps = round(args.data_passes_per_epoch * len(data_train) / args.batch_size)

	combined_train = True

	if combined_train:
		# Specs for train and eval
		train_spec = tf.estimator.TrainSpec(input_fn=data_train.input_fn)
		eval_spec = tf.estimator.EvalSpec(input_fn=data_test.input_fn, throttle_secs=10)

		for i in range(args.epochs):
			tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	else:
		for i in range(args.epochs):
			estimator.train(
				input_fn=data_train.input_fn, 
				max_steps=max_steps
			)

	result = estimator.evaluate(
		input_fn=data_test.input_fn
	)

	print(f"Accuracy: {round(result['accuracy']*100)}%")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', 			type=str, default="hosted")
	parser.add_argument('--model-dir', 			type=str, default="./output/")
	parser.add_argument('--batch-size', 		type=int, default=32)
	parser.add_argument('--embedding-width', 	type=int, default=64)
	parser.add_argument('--epochs', 			type=int, default=1)
	parser.add_argument('--data-passes-per-epoch',type=int, default=20)
	parser.add_argument('--lr', 				type=float, default=0.1)
	args = parser.parse_args()

	tf.logging.set_verbosity('INFO')
	train(args)


