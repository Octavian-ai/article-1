
import traceback
import argparse
import os.path

import tensorflow as tf
import numpy as np

from .args import get_args
from .data import GraphData
# from .model_pbt import model_fn
from .model import model_fn

from .pbt_schedule import *
from .pbt_param import *
from .pbt import Supervisor
from .estimator_worker import EstimatorWorker


def gen_param_spec(args):
	return {
		# "lr": FixedParamOf(args.lr), #LRParam,
		# "embedding_width": FixedParamOf(args.embedding_width),
		# "batch_size": lambda: IntParam(pow(10, random.uniform(0,3))),
		# "vars": VariableParam,
		"heritage": Heritage,
		"model_id": ModelId,
		"macro_step": FixedParamOf(args.macro_step),
		"micro_step": FixedParamOf(args.micro_step),
		# "n_train": FixedParamOf(None), #lambda: IntParam(random.randint(3, 10000), 1, 10000),
		# "n_val": FixedParamOf(None), #lambda: IntParam(random.randint(3, 1000), 1, 1000),
		"cluster_factor": lambda: MulParam(0.0, 0.0, 1.0),
		# "n_cluster": FixedParamOf(6),
	}

def gen_worker_init_params(args):
	person_ids = {}
	product_ids = {}

	data_train = GraphData(args, person_ids, product_ids)
	data_test  = GraphData(args, person_ids, product_ids, test=True)

	estimator_params = vars(args)
	estimator_params["n_person"] = len(person_ids)
	estimator_params["n_product"] = len(product_ids)

	worker_init_params = {
		"model_fn": model_fn, 
		"estimator_params": estimator_params, 
		"train_input_fn": lambda params: data_train.input_fn, # lambda params: lambda: data_train.gen_input_fn(args["batch_size"].value, args["n_train"].value), 
		"eval_input_fn": lambda params: data_test.input_fn,   # lambda params: lambda: data_test.gen_input_fn(args["batch_size"].value, args["n_val"].value),
		"model_dir": args.model_dir
	}

	return worker_init_params



def train(args):

	def score(worker):
		# return worker.results.get("accuracy", 0) * worker.params["n_val"].value
		return worker.results.get("accuracy", 0) * 100

	# decay_schedule(start_val=30,end_val=10)

	s = Supervisor(
		args,
		EstimatorWorker, 
		gen_worker_init_params(args), 
		gen_param_spec(args), 
		score=score,
		n_workers=args.n_workers)

	s.run(args.epochs)


if __name__ == '__main__':
	# tf.logging.set_verbosity('INFO')
	args = get_args()
	train(args)




