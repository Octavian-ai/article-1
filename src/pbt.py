
import tensorflow as tf
import random
import pickle
import pathlib
import traceback
from glob import glob
import uuid
import os
import collections

from .ploty import Ploty


FP = collections.namedtuple('FallbackParam', ['value'])


class Worker(object):
	"""Runs a PBT experiment

	Always provide a parameterless init so the Supervisor can spawn workers as needed

	"""
	def __init__(self, init_params, hyperparam_spec):
		self.count = 0
		self.id = uuid.uuid1()
		self.results = {}
		self.init_params = init_params
		self.gen_params(hyperparam_spec)

	def gen_params(self, hyperparam_spec):
		self.params = {
			k: v() for k, v in hyperparam_spec.items()
		}

	@property
	def params(self):
		pass
	
	@params.setter
	def params(self, params):
		pass
		
	def reset_count(self):
		self.count = 0
	
	def step(self, steps):
		self.count += steps
		self.do_step(steps)
	
	def do_step(self, steps):
		pass
 
	def eval(self):
		self.results = self.do_eval()
		return self.results
	
	def do_eval(self):
		pass

	def is_ready(self):
		self.count > self.params.get("micro_step", FP(1)).value * self.params.get("macro_step", FP(5)).value

	def save(self, path):
		# os.makedirs(path, exist_ok=True)
		with open(path, 'wb') as file:
			pickle.dump(self, file)

	@classmethod
	def load(cls, path, init_params):
		with open(path, 'rb') as file:
			w = pickle.load(file)
		w.init_params = init_params
		return w


	 
		
		
class Supervisor(object):
	"""Implementation of Population Based Training. Supervisor manages and optimises the experiments"""
	def __init__(self, 
				 SubjectClass, 
				 init_params,
				 hyperparam_spec, 
				 output_dir,
				 score,
				 n_workers=10, 
				 save_freq=20):

		self.SubjectClass = SubjectClass
		self.init_params = init_params
		self.hyperparam_spec = hyperparam_spec
		self.output_dir = output_dir
		self.score = score
		self.save_freq = save_freq
		self.save_counter = save_freq

		# Function or Integer supported
		if isinstance(n_workers, int) or isinstance(n_workers, float):
			self.n_workers = lambda step: round(n_workers)
		else:
			self.n_workers = n_workers

		self.fail_count = 0
		self.workers = []
		self.plot = Ploty(title='Training progress', x='Time', y="Score", output_path=output_dir)
		self.hyperplot = Ploty(title='Hyper parameters', x='Time', y="Value", output_path=output_dir)


	def save(self):
		p = f"{self.output_dir}/population"

		try:
			pathlib.Path(p).mkdir(parents=True, exist_ok=True) 
		except:
			pass

		for worker in self.workers:
			worker.save(f"{p}/worker_{worker.id}.pkl")

		tf.logging.info(f"Saved workers")

	def load(self, input_dir):
		pop_dir = f"{input_dir}/population/worker_*.pkl"
		tf.logging.info(f"Trying to load workers from {pop_dir}")

		self.workers = []
		for i in glob(pop_dir):
			tf.logging.info(f"Loading {i}")
			try:
				w = self.SubjectClass.load(i, self.init_params)
				self.workers.append(w)
				tf.logging.info(f"Loaded {w.id} {self.score(w)}")

			except Exception as e:
				print(e)

		tf.logging.info(f"Loaded workers")



	def scale_workers(self, epoch):

		stack = list(self.workers)
		random.shuffle(stack) # Tie-break randomly
		stack = sorted(stack, key=self.score)
		
		n20 = round(len(stack)*0.2)
		bottom20 = stack[:n20]

		delta = self.n_workers(epoch) - len(self.workers)

		if delta != 0:
			tf.logging.info(f"Resizing worker pool by {delta}")

		if delta < 0:
			ws = sorted(self.workers, key=self.score)
			self.workers = ws[min(-delta, len(self.workers)):]

		elif delta > 0:	
			for i in range(delta):
				additional = self.SubjectClass(self.init_params, self.hyperparam_spec)
				additional.count = random.randint(0,
					round(additional.params.get('macro_step', FP(5)).value * 0.2)
				)
				self.workers.append(additional)

		
	def exploit(self, worker):
		stack = list(self.workers)
		random.shuffle(stack) # Tie-break randomly
		stack = sorted(stack, key=self.score)
		
		n20 = round(len(stack)*0.2)
		top20 = stack[-n20:]
		bottom20 = stack[:n20]
		
		if worker in bottom20:
			mentor = random.choice(top20)
			return mentor.params
		else:
			return worker.params
		
	
	def explore(self, params):
		return {
				k:v.mutate(params.get("heat", FP(1.0)).value) for k, v in params.items()
		}
	
	def print_status(self, epoch):

		measures = {
			"score": self.score,
			# "validation": lambda i: i.results.get('val_acc', -1),
			# "train": lambda i: i.results.get('train_acc', -1)
		}
		
		for i, worker in enumerate(self.workers):
			for key, fn in measures.items():
				self.plot.add_result(epoch, fn(worker),  str(i)+key, "s", '-')
			
			
		self.plot.render()
		self.plot.save_csv()

		for key, fn in measures.items():
			vs = [fn(i) for i in self.workers]

			if len(vs) > 0:
				best = max(vs)
				worst = min(vs)
				self.hyperplot.add_result(epoch, best, f"{key}_max")
				self.hyperplot.add_result(epoch, worst, f"{key}_min")

		self.hyperplot.add_result(epoch, len(self.workers), "n_workers")

		best_worker = max(self.workers, key=self.score)

		for key, val in best_worker.params.items():
			if isinstance(val.value, int) or isinstance(val.value, float):
				self.hyperplot.add_result(epoch, val.value, f"{key}_best")



	# TODO: Make params into a virtual dictionary (and wrap .value for the caller)
	def params_equal(self, p1, p2):
		for k, v in p1.items():
			if v != p2[k]:
				return False
		return True

	def _remove_worker(self, worker, epoch):
		self.workers.remove(worker)
		self.fail_count += 1
		self.hyperplot.add_result(epoch, self.fail_count, "failed_workers")


	def step(self, epoch):
		for i in self.workers:
			try:
				steps = i.params.get("micro_step", FP(1)).value
				tf.logging.info(f"{i.id} train {steps}")
				i.step(steps)
				i.eval()
				tf.logging.info(f"{i.id} eval {self.score(i)}")
			except Exception:
				traceback.print_exc()
				self._remove_worker(i, epoch)
				continue

			
			if i.is_ready():
				i.reset_count()
				
				params = i.params
				params2 = self.exploit(i)
				
				if not self.params_equal(params, params2):
					tf.logging.info(f"Replace with exploit-explore {i.id}")
					i.params = self.explore(params2)

					try:
						i.eval()
					except Exception:
						traceback.print_exc()
						self._remove_worker(i, epoch)
						continue

		if len(self.workers) == 0:
			raise Exception("All workers failed, your model has bugs")

		self.save_counter -= 1;
		if self.save_counter <= 0:
			self.save()
			self.save_counter = self.save_freq
					
			
		
	def run(self, epochs=1000):
		for i in range(epochs):
			self.scale_workers(i)
			self.step(i)
			self.print_status(i)
			
