
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *

from .env import test_args



class WidgetTestCase(unittest.TestCase):

	def assertDictAlmostEqual(self, first, second, places, msg=None):
		for key, val in first.items():
			self.assertAlmostEqual(val, second[key], places, msg)

	def setUp(self):
		self.init_params = gen_worker_init_params(test_args)

	def test_save_load(self):
		file_path = test_args.output_dir + "worker1"

		worker = EstimatorWorker(self.init_params, pbt_param_spec)
		worker.step(20)
		worker.eval()
		worker.save(file_path)
		print(f"Worker r={worker.results}, p={worker.params}")

		worker2 = EstimatorWorker.load(file_path, self.init_params)
		print(f"Worker2 r={worker2.results}, p={worker2.params}")

		self.assertEqual(worker.results, worker2.results)
		self.assertEqual(worker.params, worker2.params)

		worker2.eval()

		self.assertDictAlmostEqual(worker.results, worker2.results, 2, "Evaluation after loading and eval should be unchanged")
		self.assertEqual(worker.params, worker2.params)

		worker2.step(20)
		worker2.eval()

		self.assertNotEqual(worker.results, worker2.results)
		self.assertEqual(worker.params, worker2.params)

if __name__ == '__main__':
	tf.logging.set_verbosity('INFO')
	unittest.main()