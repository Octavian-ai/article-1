
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
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	data_train.write_labels(model_dir, "train")
	# data_test.write_labels(model_dir, "test")


	# --------------------------------------------------------------------------
	# Build model and train
	# --------------------------------------------------------------------------

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		model_dir=model_dir,
		params={
			"lr": args.lr,
			"n_person": len(person_ids),
			"n_product": len(product_ids),
			"embedding_width": args.embedding_width
		})


	max_steps = round(
		float(args.data_passes_per_epoch * len(data_train) * args.epochs) / args.batch_size
	)

	# Let's train!
	train_spec = tf.estimator.TrainSpec(input_fn=data_train.input_fn, max_steps=max_steps)
	eval_spec = tf.estimator.EvalSpec(input_fn=data_test.input_fn, throttle_secs=10)

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

	result = estimator.evaluate(
		input_fn=data_test.input_fn
	)


	print(result)
	print(f"Accuracy: {round(result['accuracy']*100)}%")



if __name__ == '__main__':
	args = get_args()
	tf.logging.set_verbosity('INFO')
	train(args)


