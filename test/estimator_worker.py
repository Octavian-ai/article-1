
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *

from .env import *
test_args = gen_args()

file_path = test_args.output_dir + "worker1"

class EstimatorWorkerTestCase(unittest.TestCase):

	# Helpers
	def assertDictAlmostEqual(self, first, second, threshold=0.4, msg=None):
		for key, val in first.items():
			delta = abs(float(val) - float(second[key]))
			pct = delta / float(val)
			self.assertTrue(pct < threshold, key + ": " + msg)
			# self.assertAlmostEqual(val, second[key], places, key + ": " + msg)

	def vend_worker(self):
		return EstimatorWorker(self.init_params, self.hyperparam_spec)


	# Setup and teardown

	def setUp(self):
		self.init_params = gen_worker_init_params(test_args)
		self.hyperparam_spec = gen_param_spec(test_args)

	
	# ==========================================================================
	# Tests
	# ==========================================================================
	
	def test_save_load(self):
		worker1 = self.vend_worker()
		worker1.step(20)
		worker1.eval()
		worker1.save(file_path)
		worker2 = EstimatorWorker.load(file_path, self.init_params)

		self.assertEqual(worker1.results, worker2.results)
		self.assertEqual(worker1.params, worker2.params)

		worker2.eval()

		self.assertDictAlmostEqual(worker1.results, worker2.results, msg="Evaluation after loading and eval should be unchanged")
		self.assertEqual(worker1.params, worker2.params)

	def test_param_copy(self):
		worker1 = self.vend_worker()
		worker1.step(20)
		worker1.eval()

		worker2 = self.vend_worker()
		worker2.params = worker1.params
		worker2.eval()

		# self.assertGreaterEqual(worker2.results["accuracy"], worker1.results["accuracy"])
		self.assertDictAlmostEqual(worker1.results, worker2.results, msg="Evaluation after param copy should be the same")
		



if __name__ == '__main__':
	tf.logging.set_verbosity('INFO')
	unittest.main()


