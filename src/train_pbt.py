
import traceback
import argparse

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


pbt_param_spec = {
	"lr": LRParam,
	"embedding_width": lambda: IntParam(pow(10, random.uniform(0,2.5))),
	"batch_size": lambda: IntParam(pow(10, random.uniform(0,3))),
	# "vars": VariableParam,
	"heritage": Heritage,
	"model_id": ModelId,
	"macro_step": IntParamOf(5),
	"micro_step": lambda: IntParam(random.randint(3, 300)),
	"heat": lambda: MulParam(random.uniform(0.3,5)),
	"n_train": lambda: IntParam(random.randint(3, 10000), 1, 10000),
	"n_val": lambda: IntParam(random.randint(3, 1000), 1, 1000),
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
		"train_input_fn": lambda args: lambda: data_train.gen_input_fn(args["batch_size"].value, args["n_train"].value), 
		"eval_input_fn": lambda args: lambda: data_test.gen_input_fn(args["batch_size"].value, args["n_val"].value),
		"model_dir": args.output_dir + "checkpoint/"
	}

	return worker_init_params



def train(args):

	worker_init_params = gen_worker_init_params(args)
	
	def score(worker):
		return worker.results.get("accuracy", 0) * worker.params["n_val"].value

	s = Supervisor(
		EstimatorWorker, 
		worker_init_params, 
		pbt_param_spec, 
		args.output_dir, 
		score=score,
		n_workers=decay_schedule(start_val=40))

	s.run(args.epochs)


if __name__ == '__main__':
	# tf.logging.set_verbosity('INFO')
	args = get_args()
	train(args)




