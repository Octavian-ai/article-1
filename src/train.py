
import traceback
import os.path
import csv
from datetime import datetime

import tensorflow as tf
import numpy as np

from .args import get_args
from .data import GraphData
from .model import model_fn


def train(args):

	# Make our folders
	data_method = "random_walk" if args.use_random_walk else "random_rows"
	model_dir = os.path.join(args.output_dir, data_method, str(datetime.now()))
	os.makedirs(model_dir, exist_ok=True)

	# --------------------------------------------------------------------------
	# Load data
	# --------------------------------------------------------------------------
	
	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_eval  = GraphData(args, person_ids, product_ids, test=True)

	# --------------------------------------------------------------------------
	# Build model and train
	# --------------------------------------------------------------------------

	params = vars(args)
	params["n_person"]  = len(person_ids)
	params["n_product"] = len(product_ids)

	# Determined through grid search and PBT
	params["embedding_width"] = 20 if args.use_random_walk else 82

	estimator = tf.estimator.Estimator(
		model_fn  = model_fn,
		model_dir = model_dir,
		params    = params
	)

	train_spec = tf.estimator.TrainSpec(input_fn=data_train.input_fn, max_steps=args.max_steps)
	eval_spec  = tf.estimator.EvalSpec(input_fn=data_eval.input_fn, steps=None)
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
	args = get_args()
	tf.logging.set_verbosity('INFO')

	for i in range(args.runs):
		train(args)


