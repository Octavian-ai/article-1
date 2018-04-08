
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *

from .env import test_args, test_dir



class WidgetTestCase(unittest.TestCase):
	def setUp(self):
		self.init_params = gen_worker_init_params(test_args)

	def test_save_load(self):
		file_path = test_dir + "/worker1"

		worker = EstimatorWorker(self.init_params, pbt_param_spec)
		worker.step(20)
		worker.eval()

		worker.save(file_path)
		worker2 = EstimatorWorker.load(file_path, self.init_params)

		self.assertEqual(worker.results, worker2.results)
		self.assertEqual(worker.params, worker2.params)

if __name__ == '__main__':
	tf.logging.set_verbosity('INFO')
	unittest.main()