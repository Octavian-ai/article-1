
import traceback
import argparse

import tensorflow as tf
import numpy as np

from .args import get_args
from .data import GraphData
from .model_pbt import model_fn

from .pbt_param import *
from .estimator_worker import *

pbt_param_spec = {
	"lr": LRParam,
	"embedding_width": lambda: IntParam(pow(10, random.uniform(0,2.5)), 1, 1000),
	"batch_size": lambda: IntParam(pow(10, random.uniform(0,3)), 1, 1000),
	# "vars": VariableParam,
	"heritage": Heritage,
	"model_id": ModelId
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
		"train_input_fn": lambda args: lambda: data_train.gen_input_fn(args["batch_size"].value), 
		"eval_input_fn": lambda args: lambda: data_test.gen_input_fn(args["batch_size"].value),
		"model_dir": args.output_dir + "checkpoint/"
	}

	return worker_init_params



def train(args):

	worker_init_params = gen_worker_init_params(args)
	
	def score(worker):
		return worker.results.get("accuracy", 0)

	s = Supervisor(
		EstimatorWorker, 
		worker_init_params, 
		pbt_param_spec, 
		args.output_dir, 
		score,
		micro_step=300,
		macro_step=3)

	s.run(args.epochs)


if __name__ == '__main__':
	# tf.logging.set_verbosity('INFO')
	args = get_args()
	train(args)




