
import tensorflow as tf


features = {
	"person_id":  tf.placeholder(dtype=tf.int32, shape=[None, 1]),
	"product_id": tf.placeholder(dtype=tf.int32, shape=[None, 1]),
}

serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
    features
)