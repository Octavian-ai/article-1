
from src.args import get_args 

def gen_args(bucket=None, gcs_dir=None):
	args = [
		"--output-dir=./output_test",
		"--model-dir=./output_test/checkpoint",
		"--shuffle-batch=False",
	]

	if bucket is not None:
		args.append("--bucket=" + bucket)

	if gcs_dir is not None:
		args.append("--gcs-dir=" + gcs_dir)

	return get_args(args)