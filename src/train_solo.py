
import traceback
import argparse
import pickle
import os.path
import csv

import tensorflow as tf
import numpy as np

from .args import get_args
from .data import GraphData
from .model import model_fn


def train(args):

	# Make our folders
	suffix = f"{args.embedding_width}"
	model_dir = os.path.join(args.output_dir, suffix)
	os.makedirs(model_dir, exist_ok=True)

	# --------------------------------------------------------------------------
	# Load data
	# --------------------------------------------------------------------------
	
	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_eval  = GraphData(args, person_ids, product_ids, test=True)

	data_train.write_labels(model_dir, "train")
	# data_eval.write_labels(model_dir, "test")


	# --------------------------------------------------------------------------
	# Build model and train
	# --------------------------------------------------------------------------

	params = vars(args)
	params["n_person"]  = len(person_ids)
	params["n_product"] = len(product_ids)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=model_dir,
		params=params
	)


	if args.mode == 'all' or args.mode == 'train':

		max_steps = round(
			float(args.data_passes_per_epoch * len(data_train) * args.epochs) / args.batch_size
		)
		tf.logging.info(f"Training for max_steps {max_steps}")

		train_spec = tf.estimator.TrainSpec(input_fn=data_train.input_fn_walk, max_steps=max_steps)
		eval_spec = tf.estimator.EvalSpec(input_fn=data_eval.input_fn_walk, throttle_secs=30)

		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	

	if args.mode == 'all' or args.mode == 'evaluate':
		result = estimator.evaluate(
			input_fn=data_eval.input_fn
		)
		print(result)



	# --------------------------------------------------------------------------
	# Generate predictions then save them to a CSV
	# --------------------------------------------------------------------------

	if args.mode == 'all' or args.mode == 'predict':
		preds = estimator.predict(data_eval.input_fn)
		with open(os.path.join(args.output_dir, "predictions.csv"), 'w') as file:

			writer = None

			for i in preds:
				# Let's peek to get the keys
				if writer is None:
					writer = csv.DictWriter(file, fieldnames=i.keys())
					writer.writeheader()

				writer.writerow(i)


if __name__ == '__main__':
	args = get_args()
	tf.logging.set_verbosity('INFO')
	train(args)


