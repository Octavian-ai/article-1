
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

# Reduce code duplication throughout these methods
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
				review.score as review_score
		"""

		self.query_params = {
			"dataset_name": "article_1",
			"test": test
		}

		self.settings = settings[args.database]

		driver = GraphDatabase.driver(
			self.settings["neo4j_url"], 
			auth=(self.settings["neo4j_user"], self.settings["neo4j_password"]))

		self.ids = {
			"person": person_ids,
			"product": product_ids
		}


		def data_to_vec(i):
			return ({
					"person_id": self._get_index(i, "person"),
					"product_id": self._get_index(i, "product"),
				}, 
				i["review_score"]	
			)

		with driver.session() as session:
			self.raw_data = session.run(self.query, **self.query_params).data()
			data = [ data_to_vec(i) for i in self.raw_data ]

			# Remove any ordering biases from the database
			random.seed(123)
			random.shuffle(data)

			# Index the rows by person_id and _product_id
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
	

	def gen_walk(self, batch_size):
		"""Generate random walks across our graph."""

		def next_noun(prev):
			found_prev = False
			for noun in nouns:
				if noun == prev:
					found_prev = True
				elif found_prev:
					return noun

			# Loop back to start
			return nouns[0]

		for noun in nouns:
			for obj_id in self.indexed_data[noun].keys():
				rows = self.indexed_data[noun][obj_id]

				if len(rows) > 0:
					batch = []
					
					batch.append(random.choice(rows))
					noun_to_join = next_noun(noun)

					while len(batch) < batch_size:
						next_id = batch[-1][0][noun_to_join+"_id"]
						next_rows = self.indexed_data[noun_to_join].get(next_id, [])

						if len(next_rows) > 0:
							batch.append(random.choice(next_rows))
							noun_to_join = next_noun(noun_to_join)
						else:
							break

					# If we somehow indexed into a dead end above (highly unlikely) then
					# pad the rest of the batch with random rows
					while len(batch) < batch_size:
						batch.append(random.choice(self.data))

					for b in batch:
						yield b

	
	

	def gen_dataset_walk(self, batch_size):
		return tf.data.Dataset.from_generator(
			lambda: self.gen_walk(batch_size),
			self.dataset_dtype,
			self.dataset_size
		).batch(batch_size)


	def gen_dataset_rand(self, batch_size):
		return tf.data.Dataset.from_generator(
			lambda: (i for i in self.data),
			self.dataset_dtype,
			self.dataset_size
		).shuffle(len(self)).batch(batch_size)


	# This is a little syntactic sugar so the caller can pass input_fn directly into Estimator.train()
	@property
	def input_fn(self):
		if self.args.use_random_walk:
			return self.input_fn_walk
		else:
			return self.input_fn_rand

	@property
	def input_fn_rand(self):
		return lambda: self.gen_dataset_rand(self.args.batch_size)

	@property
	def input_fn_walk(self):
		return lambda: self.gen_dataset_walk(self.args.batch_size)

	@property
	def dataset_dtype(self):
		return (
			{
				"person_id": tf.int32,
				"product_id": tf.int32,
			}, 
			tf.float32
		)

	@property
	def dataset_size(self):
		return (
			{
				"person_id": tf.TensorShape([]),
				"product_id": tf.TensorShape([]),
			}, 
			tf.TensorShape([])
		)



	# --------------------------------------------------------------------------
	# Utilities
	# --------------------------------------------------------------------------
	
	@property
	def n_person(self):
		return len(self.ids["person"])

	@property
	def n_product(self):
		return len(self.ids["product"])

	def __len__(self):
		return len(self.data)


