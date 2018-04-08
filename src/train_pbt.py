
import traceback
import argparse

import tensorflow as tf
import numpy as np

from .data import GraphData
from .model_pbt import model_fn

from .pbt_param import *
from .estimator_worker import *

pbt_param_spec = {
	"lr": LRParam,
	"embedding_width": lambda: IntParam(pow(10, random.uniform(0,2.5)), 1, 1000),
	"vars": VariableParam,
	"heritage": Heritage
}

def gen_worker_init_params(args):
	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	estimator_params = {
		"n_person": len(person_ids),
		"n_product": len(product_ids)
	}

	worker_init_params = {
		"model_fn": model_fn, 
		"estimator_params": estimator_params, 
		"train_input_fn": data_train.input_fn, 
		"eval_input_fn": data_test.input_fn
	}

	return worker_init_params


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', 			type=str, default="hosted")
	parser.add_argument('--output-dir', 		type=str, default="./output-pbt/")
	parser.add_argument('--batch-size', 		type=int, default=32)
	parser.add_argument('--epochs', 			type=int, default=30)
	parser.add_argument('--data-passes-per-epoch',type=int, default=2)
	return parser.parse_args()


def train(args):

	worker_init_params = gen_worker_init_params(args)
	
	def score(worker):
		return worker.results["accuracy"]

	s = Supervisor(EstimatorWorker, worker_init_params, pbt_param_spec, args.output_dir, score)

	s.run(args.epochs)


if __name__ == '__main__':
	tf.logging.set_verbosity('INFO')
	args = get_args()
	train(args)




