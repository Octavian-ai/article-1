
import traceback
import argparse

import tensorflow as tf
import numpy as np

from data import GraphData
from model import model_fn


def train(args):

	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	suffix = f"{len(person_ids)}-{len(product_ids)}-{args.embedding_width}"

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=args.model_dir + suffix,
		params={
			"lr": args.lr,
			"n_person": len(person_ids),
			"n_product": len(product_ids),
			"embedding_width": args.embedding_width
		})

	combined_train = True

	if combined_train:
		# Specs for train and eval
		train_spec = tf.estimator.TrainSpec(input_fn=data_train.input_fn)
		eval_spec = tf.estimator.EvalSpec(input_fn=data_test.input_fn, throttle_secs=10)

		for i in range(args.epochs):
			tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	else:
		max_steps = round(args.data_passes_per_epoch * len(data_train) / args.batch_size)

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


