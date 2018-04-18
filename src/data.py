
import tensorflow as tf
import numpy as np
import os.path
import random
import json
from neo4j.v1 import GraphDatabase, Driver

# After fighting with gcloud, I dumped this here. Ideally, package elsewhere
settings = {
	"hosted": {
		"neo4j_url": "bolt://b2355e7e.databases.neo4j.io",
		"neo4j_user": "readonly",
		"neo4j_password": "OS5jLkVsOUZCVTdQOU5PazBo"
	},
	"local": {
		"neo4j_url": "bolt://localhost",
	    "neo4j_user": "neo4j",
	    "neo4j_password": "neo4j"
	}
}

nouns = ["person", "product"]

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
				review.score as review_score
			LIMIT 3
		"""

		self.query_params = {
			"dataset_name": "article_1",
			"test": test
		}

		self.settings = settings[args.database]

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


		def data_to_vec(i):
			return ({
					"person": {
						"id": self._get_index(i, "person"),
						"style": i["person_style"],
					},
					"product": {
						"id": self._get_index(i, "product"),
						"style": i["product_style"],
					}, 
					"review_score": i["review_score"],
				}, 
				(
					i["review_score"]	
				)
			)

			# return (
			# 			( 
			# 				self._get_index(i, "person"), 
			# 				self._get_index(i, "product"),
			# 				i["person_style"],
			# 		  		i["product_style"],
			# 		  		i["review_score"], # For prediction debugging convenience. Not used for training
			# 			), 
			# 		  	( 
			# 		  		i["review_score"]
			# 		  	)
			# 		 )


		with driver.session() as session:
			self.raw_data = session.run(self.query, **self.query_params).data()
			data = [ data_to_vec(i) for i in self.raw_data ]

			random.seed(123)
			random.shuffle(data)

			self.indexed_data = {}
			for noun in nouns:
				self.indexed_data[noun] = {
					self._uuid_to_index(k, self.ids[noun]): [
						data_to_vec(i) for i in self.raw_data if i[noun+"_id"] == k
					] 
					for k in self.ids[noun]
				}


			self.data = data


	


	# --------------------------------------------------------------------------
	# Input functions
	# --------------------------------------------------------------------------
	

	def generate_walks(self):

		def next_noun(prev):
			for noun in nouns:
				if noun != prev:
					return noun

		for noun in nouns:
			for i in self.indexed_data[noun]:
				batch = []
				noun_to_join = next_noun(noun)
				batch.append(random.choice(i))

				while len(batch) < self.args.batch_size:
					
					next_id = batch[-1][noun_to_join]["id"]
					next_rows = self.indexed_data[noun_to_join].get(next_id, [])

					if len(next_rows) > 0:
						batch.append(random.choice(next_rows))
					else:
						# Start again
						noun_to_join = next_noun(noun)
						batch.append(random.choice(i))

				yield batch

	
	
	
	@property
	def input_fn_walk(self):
		return lambda: tf.data.Dataset.from_generator(
			lambda: self.generate_walks(),
			self.dataset_dtype,
			self.dataset_size
		)
	


	def gen_input_fn(self, batch_size=None, limit=None):

		if limit is None:
			limit = len(self.data)

		if batch_size is None:
			batch_size = self.args.batch_size

		d = tf.data.Dataset.from_generator(
			lambda: (i for i in self.data[:limit]),
			self.dataset_dtype,
			self.dataset_size
		)

		# d = d.apply(tf.contrib.data.shuffle_and_repeat(len(self), self.args.data_passes_per_epoch))
		d = d.shuffle(len(self), reshuffle_each_iteration=self.args.shuffle_batch)
		d = d.repeat(self.args.data_passes_per_epoch)
		d = d.batch(batch_size)

		return d

	# This is a little syntactic sugar so the caller can pass input_fn directly into Estimator.train()
	@property
	def input_fn(self):
		return lambda: self.gen_input_fn()

	@property
	def dataset_dtype(self):
		return (
			{
				"person": {
					"id": tf.int32,
					"style": tf.float32
				},
				"product": {
					"id": tf.int32,
					"style": tf.float32
				}, 
				"review_score": tf.float32
			}, 
			(tf.float32)
		)

	@property
	def dataset_size(self):
		return (
			{
				"person": {
					"id": tf.TensorShape([]),
					"style": tf.TensorShape([6]),
				},
				"product": {
					"id": tf.TensorShape([]),
					"style": tf.TensorShape([6]),
				}, 
				"review_score": tf.TensorShape([]) 
			}, 
			( tf.TensorShape([]) )
		)


	# --------------------------------------------------------------------------
	# Utilities
	# --------------------------------------------------------------------------
	
	@property
	def n_person(self):
		return len(self.person_ids)

	@property
	def n_product(self):
		return len(self.product_ids)

	def __len__(self):
		return len(self.data)

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

				return "{}\t{}\n".format(cls, idd)
			else:
				return "\t\n"


		for noun in nouns:
			ordered = {}

			for i in self.raw_data:
				ordered[self._get_index(i, noun)] = i
				
			with open(os.path.join(output_dir, prefix+"_"+noun+"_labels.tsv"), 'w') as label_file:
				label_file.write(header)

				for i in range(len(ordered)):
					label_file.write(format_row(i, ordered, noun))


		# Easier than the TF many lines of setup
		with open(os.path.join(output_dir, "projector_config.pbtext"), 'w') as config_file:
			config_file.write("embeddings {")

			for noun in nouns:
				config_file.write(" tensor_name: '"+noun+"'")
				config_file.write("  metadata_path: './"+prefix+"_"+noun+"_labels.tsv'")

			config_file.write("}")




