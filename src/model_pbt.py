import tensorflow as tf
import numpy as np

from .model import model_fn as model_fn_base

from .estimator_worker import gen_scaffold

# This wrapper just adds the scaffolding
# TODO: Make scaffolding come from params
def model_fn(features, labels, mode, params):

	spec = model_fn_base(features, labels, mode, params)

	# Wrap this up into an EstimatorSpec for Tensorflow to train using high level API
	return tf.estimator.EstimatorSpec(
		mode, 
		loss=spec.loss, 
		train_op=spec.train_op, 
		eval_metric_ops=spec.eval_metric_ops,
		scaffold=gen_scaffold(params))


