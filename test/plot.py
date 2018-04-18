
import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *

from .env import *


class PlotTestCase(unittest.TestCase):

	def test_basics(self):
		test_args = gen_args()
		ploty = Ploty(test_args, 'test_basics')

		try:
			os.remove(ploty.file_path)
		except FileNotFoundError:
			pass

		for i in range(10):
			ploty.add_result(i, i, 'id')

		ploty.write()

		self.assertTrue(os.path.isfile(ploty.file_path))

	def test_gcs(self):
		test_args = gen_args('octavian-test', 'unittest')
		ploty = Ploty(test_args, 'test_gcs')

		try:
			os.remove(ploty.file_path)
		except FileNotFoundError:
			pass

		for i in range(10):
			ploty.add_result(i, i, 'id')

		
		ploty.write()

		self.assertTrue(os.path.isfile(ploty.file_path))





if __name__ == '__main__':
	tf.logging.set_verbosity('INFO')
	unittest.main()


