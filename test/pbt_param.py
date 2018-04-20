
import unittest
import os.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *



class PbtParamTestCase(unittest.TestCase):

	
	def test_int_param(self):
		p = IntParam(30,1,100)

		for i in range(20):
			self.assertIsInstance(p.value, int)
			p = p.mutate()


if __name__ == '__main__':
	unittest.main()


