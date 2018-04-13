
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', 				type=str, default="hosted")
	parser.add_argument('--output-dir', 			type=str, default="./output/")
	parser.add_argument('--mode', 					type=str, choices=['all', 'train', 'predict', 'evaluate'], default='all')
	
	parser.add_argument('--epochs', 				type=int, default=20)
	parser.add_argument('--data-passes-per-epoch',	type=int, default=20)
	parser.add_argument('--shuffle-batch',			type=bool, default=True)
	parser.add_argument('--batch-size', 			type=int, default=32)
	parser.add_argument('--embedding-width', 		type=int, default=82)
	parser.add_argument('--n_cluster', 				type=int, default=6)
	parser.add_argument('--lr',						type=float, default=0.103557887)
	parser.add_argument('--cluster-factor',			type=float, default=0.3)

	

	return parser.parse_args()
