
import traceback
import argparse

import tensorflow as tf
import numpy as np

from data import GraphData
from model_pbt import model_fn

from pbt_param import *
from pbt_worker import *



def train(args):

	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	estimator_params = {
		"n_person": len(person_ids),
		"n_product": len(product_ids)
	}

	param_spec = {
		"lr": LRParam,
		"embedding_width": lambda: IntParam(pow(10, random.uniform(0,2.5)), 1, 1000),
		"vars": VariableParam,
		"heritage": Heritage
	}

	EstimatorWorker = MetaEstimatorWorker(
		model_fn, estimator_params, data_train.input_fn, data_test.input_fn
	)

	def score(worker):
		return worker.results["accuracy"]

	s = Supervisor(EstimatorWorker, param_spec, args.output_dir, score)

	s.run(args.epochs)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', 			type=str, default="hosted")
	parser.add_argument('--output-dir', 		type=str, default="./output-pbt/")
	parser.add_argument('--batch-size', 		type=int, default=32)
	parser.add_argument('--epochs', 			type=int, default=30)

	parser.add_argument('--data-passes-per-epoch',type=int, default=2)
	args = parser.parse_args()

	tf.logging.set_verbosity('INFO')
	train(args)




