
import argparse

def get_args(args=None):
	parser = argparse.ArgumentParser()

	# Data
	parser.add_argument('--database', 				type=str,  default="hosted")
	
	# Storage
	parser.add_argument('--output-dir', 			type=str,  default="./output")
	parser.add_argument('--model-dir',	 			type=str,  default="./output/checkpoint")

	parser.add_argument('--disable-random-walks',	action='store_false',dest="use_random_walk")
	
	# Training
	parser.add_argument('--max-steps', 				type=int,  default=30000)
	parser.add_argument('--runs', 					type=int,  default=1)
	parser.add_argument('--batch-size', 			type=int,  default=32)
	parser.add_argument('--lr',						type=float, default=0.103557887)

	return parser.parse_args(args)
