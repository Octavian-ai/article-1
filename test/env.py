
import collections

Args = collections.namedtuple('Args', [
	'batch_size', 'database', 'data_passes_per_epoch', 
	"output_dir", "shuffle_batch"])

test_args = Args(32, 'hosted', 1, "./output_test/", False)