
import traceback
import argparse
import pickle
import os.path

import tensorflow as tf
import numpy as np

from .args import get_args
from .data import GraphData
from .model import model_fn


def train(args):

	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	suffix = f"{len(person_ids)}-{len(product_ids)}-{args.embedding_width}"
	output_dir = args.output_dir + suffix

	os.makedirs(output_dir, exist_ok=True)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=output_dir,
		params={
			"lr": args.lr,
			"n_person": len(person_ids),
			"n_product": len(product_ids),
			"embedding_width": args.embedding_width
		})

	combined_train = True

	if combined_train:

		max_steps = round(
			float(args.data_passes_per_epoch * len(data_train) * args.epochs) / args.batch_size
		)

		# Specs for train and eval
		train_spec = tf.estimator.TrainSpec(input_fn=data_train.input_fn, max_steps=max_steps)
		eval_spec = tf.estimator.EvalSpec(input_fn=data_test.input_fn, throttle_secs=10)

		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	else:

		max_steps = round(
			float(args.data_passes_per_epoch * len(data_train)) / args.batch_size
		)

		for i in range(args.epochs):
			estimator.train(
				input_fn=data_train.input_fn,
				max_steps=max_steps
			)

	result = estimator.evaluate(
		input_fn=data_test.input_fn
	)


	print(result)
	print(f"Accuracy: {round(result['accuracy']*100)}%")



if __name__ == '__main__':
	args = get_args()
	tf.logging.set_verbosity('INFO')
	train(args)


