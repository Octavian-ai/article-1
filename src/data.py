
import tensorflow as tf
import numpy as np

import json
from neo4j.v1 import GraphDatabase, Driver


class GraphData(object):

	def __init__(self, args, person_ids, product_ids, test=False):
		self.args = args
		
		self.query = """
			MATCH p=
				(person:PERSON) 
					-[:WROTE]-> 
				(review:REVIEW {dataset_name:{dataset_name}, test:{test}}) 
					-[:OF]-> 
				(product:PRODUCT)
			RETURN person.id as person_id, product.id as product_id, review.score as y
		"""

		self.query_params = {
			"dataset_name": "article_1",
			"test": test
		}

		with open('./settings.json') as f:
			self.settings = json.load(f)[args.database]

		driver = GraphDatabase.driver(
			self.settings["neo4j_url"], 
			auth=(self.settings["neo4j_user"], self.settings["neo4j_password"]))

		self.person_ids = person_ids
		self.product_ids = product_ids

		def uuid_to_index(uuid, db):
			if uuid not in db:
				db[uuid] = len(db)

			return db[uuid]


		with driver.session() as session:
			raw_data = session.run(self.query, **self.query_params).data()
			data = [ ((uuid_to_index(i["person_id"],  self.person_ids), 
				       uuid_to_index(i["product_id"], self.product_ids)), i["y"]) for i in raw_data]

			tf.logging.info(f"Data loaded, got {len(data)} rows, {len(self.person_ids)} person nodes, {len(self.product_ids)} product nodes")
			
			scores = [i["y"] for i in raw_data]
			tf.logging.info(f"Histogram: {np.histogram(scores)}")

			self.data = data

	@property
	def n_person(self):
		return len(self.person_ids)

	@property
	def n_product(self):
		return len(self.product_ids)

	def __len__(self):
		return len(self.data)

	def gen_input_fn(self):
		def gen():
			return (i for i in self.data)

		d = tf.data.Dataset.from_generator(
			gen,
			((tf.int32, tf.int32), tf.float32),
			((tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([]))
		)

		# d = d.apply(tf.contrib.data.shuffle_and_repeat(len(self), self.args.data_passes_per_epoch))
		d = d.shuffle(len(self), reshuffle_each_iteration=self.args.shuffle_batch)
		d = d.repeat(self.args.data_passes_per_epoch)
		d = d.batch(self.args.batch_size)

		return d

	# This is a little syntactic sugar so the caller can pass input_fn directly into Estimator.train()
	@property
	def input_fn(self):
		return lambda: self.gen_input_fn()
	



