
import tensorflow as tf
import numpy as np
import os.path
import random
import json
from neo4j.v1 import GraphDatabase, Driver


class GraphData(object):

	def _uuid_to_index(self, uuid, db):
		if uuid not in db:
			db[uuid] = len(db)

		return db[uuid]

	def _get_index(self, row, noun):
		return self._uuid_to_index(row[noun+"_id"],  self.ids[noun])


	def __init__(self, args, person_ids, product_ids, test=False):
		self.args = args
		
		self.query = """
			MATCH p=
				(person:PERSON) 
					-[:WROTE]-> 
				(review:REVIEW {dataset_name:{dataset_name}, test:{test}}) 
					-[:OF]-> 
				(product:PRODUCT)
			RETURN 
				person.id as person_id, 
				product.id as product_id, 
				person.style_preference as person_style,
				product.style as product_style,
				review.score as y
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

		# Let's start reducing copy-pasted logic
		self.ids = {
			"person": person_ids,
			"product": product_ids
		}


		with driver.session() as session:
			self.raw_data = session.run(self.query, **self.query_params).data()
			data = [ (
						(self._get_index(i, "person"), self._get_index(i, "product")), 
					  	(i["y"])
					 ) for i in self.raw_data ]

			tf.logging.info(f"Data loaded, got {len(data)} rows, {len(self.person_ids)} person nodes, {len(self.product_ids)} product nodes")
			
			scores = [i["y"] for i in self.raw_data]
			tf.logging.info(f"Histogram: {np.histogram(scores)}")

			random.seed(123)
			random.shuffle(data)

			self.data = data

	def write_labels(self, output_dir, prefix):

		nouns = ["product", "person"]
		header = "Class\tId\n"

		def format_row(index, db, noun):
			if index in db:
				style = db[index][noun+"_style"]
				idd = db[index][noun+"_id"]

				# One higher than we'd otherwise output
				cls = len(style)

				for idx, val in enumerate(style):
					if val == 1.0:
						cls = idx
						break

				return f"{cls}\t{idd}\n"
			else:
				return "\t\n"


		for noun in nouns:
			ordered = {}

			for i in self.raw_data:
				ordered[self._get_index(i, noun)] = i
				
			with open(os.path.join(output_dir, f"{prefix}_{noun}_labels.tsv"), 'w') as label_file:
				label_file.write(header)

				for i in range(len(ordered)):
					label_file.write(format_row(i, ordered, noun))


		# Easier than the TF many lines of setup
		with open(os.path.join(output_dir, "projector_config.pbtext"), 'w') as config_file:
			config_file.write("embeddings {")

			for noun in nouns:
				config_file.write(f" tensor_name: '{noun}'")
				config_file.write(f"  metadata_path: './{prefix}_{noun}_labels.tsv'")

			config_file.write("}")


	@property
	def n_person(self):
		return len(self.person_ids)

	@property
	def n_product(self):
		return len(self.product_ids)

	def __len__(self):
		return len(self.data)

	def gen_input_fn(self, batch_size=None, limit=None):

		if limit is None:
			limit = len(self.data)

		def gen():
			return (i for i in self.data[:limit])

		d = tf.data.Dataset.from_generator(
			gen,
			((tf.int32, tf.int32), tf.float32),
			((tf.TensorShape([]), tf.TensorShape([])), tf.TensorShape([]))
		)

		# d = d.apply(tf.contrib.data.shuffle_and_repeat(len(self), self.args.data_passes_per_epoch))
		d = d.shuffle(len(self), reshuffle_each_iteration=self.args.shuffle_batch)
		d = d.repeat(self.args.data_passes_per_epoch)

		if batch_size is None:
			batch_size = self.args.batch_size

		d = d.batch(batch_size)

		return d

	# This is a little syntactic sugar so the caller can pass input_fn directly into Estimator.train()
	@property
	def input_fn(self):
		return lambda: self.gen_input_fn()




