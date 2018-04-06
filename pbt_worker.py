
import tensorflow as tf
import numpy as np

from pbt import *
from pbt_param import *

"""### Train the model"""

def resize_and_load(var, val, sess):
  o_shape = var.get_shape().as_list()
  i_shape = list(val.shape)
        
  if o_shape != i_shape:
    resize_dim = 1 # may not always hold true, assumption for now
    delta = o_shape[resize_dim] - i_shape[resize_dim]
    
    if delta != 0:
      tf.logging.info(f"reshape var {var.name} by {delta}")

    if delta < 0:
      val = val[:,:o_shape[1]]
    elif delta > 0:
      val = np.pad(val, ((0,0),(0, delta)), 'mean')

    v.load(val, self.sess)


def gen_scaffold(params):
  def init_fn(scaffold, session):
    # self.sess.run(tf.global_variables_initializer())
    # self.sess.run(tf.local_variables_initializer())

    vs = params["vars"].value

    if vs is not None: 
      for var in tf.trainable_variables():
        if var.name in vs:

          val = vs[var.name]
          resize_and_load(var, val, session)

  return tf.train.Scaffold(init_fn=init_fn)


class MetricHook(tf.train.SessionRunHook):
  def __init__(self, metrics, cb, key=0):
    self.metrics = metrics
    self.key = key
    self.readings = []

  def before_run(self, run_context):
    print("before_run", run_context.original_args)
    # if self.metrics is not None:
    return run_context.original_args

  def after_run(self, run_context, run_values):
    print("after run", run_values)
    if run_values.results is not None:
      self.readings.append(run_values.results[self.key][1])

  def end(self, session):
    if len(self.readings) > 0:
      self.cb(np.average(self.readings))
      self.readings.clear()

def MetaEstimatorWorker(model_fn, estimator_params, train_input_fn, eval_input_fn):

  class EstimatorWorker(Worker):
    
    def __init__(self):
      super().__init__()
      self._params = {}
      # self._results = {}

      # def cb(accuracy):
      #   self._results["accuracy"] = accuracy

      # eval_hooks = [
      #   MetricHook(None, cb)
      # ]

      # self.train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=eval_hooks)
      # self.eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=10)
      
      self.setup_estimator()


    def setup_estimator(self):
      params = {**estimator_params, **self._params}

      self.estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params
      )

      
    @property
    def params(self):
      var_names = self.estimator.get_variable_names()
      vals = {k:self.estimator.get_variable_value(k) for k in var_names}
      self._params["vars"] = VariableParam(vals)

      return self._params;
      
    @params.setter
    def params(self, value):
      self._params = value;
      self.setup_estimator()
      tf.logging.info(f"Setup {self.id}")

      # Maybe thanks to scaffold nothing needed here, it'll pick up changes on next run
      # ????????
      # self.setup()
      
    def do_step(self, steps):
      for i in range(steps):
        # tf.estimator.train_and_evaluate(self.estimator, self.train_spec, self.eval_spec)
        self.estimator.train(train_input_fn, steps=steps)
      
    def do_eval(self):
      return self.estimator.evaluate(eval_input_fn)


    # Hooks for Pickle
    def __getstate__(self):
      return {
        "params":   self.params,
        "count":    self.count,
        "results":  self.results,
        "id":       self.id
      }

    def __setstate__(self, state):
      self.id       = state.get("id", uuid.uuid1())
      self.count    = state.get("count", 0)
      self.results  = state.get("results", {})
      self.params   = state.get("params", {})



  return EstimatorWorker


  
  