
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', 				type=str, default="hosted")
	parser.add_argument('--output-dir', 			type=str, default="./output")
	parser.add_argument('--mode', 					type=str, choices=['all', 'train', 'predict', 'evaluate'], default='all')
	
	parser.add_argument('--epochs', 				type=int, default=40)
	parser.add_argument('--micro-step', 			type=int, default=1000)
	parser.add_argument('--macro-step', 			type=int, default=5)

	parser.add_argument('--data-passes-per-epoch',	type=int, default=20)
	parser.add_argument('--shuffle-batch',			type=bool, default=True)
	parser.add_argument('--batch-size', 			type=int, default=32)
	parser.add_argument('--embedding-width', 		type=int, default=82)
	parser.add_argument('--n-clusters', 			type=int, default=6)
	parser.add_argument('--n-workers', 				type=int, default=10)

	parser.add_argument('--lr',						type=float, default=0.103557887)
	parser.add_argument('--cluster-factor',			type=float, default=0.0)

	return parser.parse_args()
