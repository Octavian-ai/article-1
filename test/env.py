
import collections

Args = collections.namedtuple('Args', ['batch_size', 'database', 'data_passes_per_epoch'])

test_args = Args(32, 'hosted', 1)
test_dir = "./output"