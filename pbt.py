
import tensorflow as tf
import random
import pickle
import pathlib
from glob import glob
import uuid

from ploty import Ploty


class Worker(object):
	"""Runs a PBT experiment"""
	def __init__(self):
		self.count = 0
		self.id = uuid.uuid1()
		self.results = {}
	
	@property
	def params(self):
		pass
	
	@params.setter
	def params(self, params):
		pass
		
	def resetCount(self):
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

	def save(self, path):
		file = open(path, 'wb')
		pickle.dump(self, file)

	@classmethod
	def load(cls, path):
		file = open(path, 'rb')
		return pickle.load(file)


	 
		
		
class Supervisor(object):
	"""Implementation of Population Based Training. Supervisor manages and optimises the experiments"""
	def __init__(self, 
					SubjectClass, 
					paramSpec, 
					output_dir,
					score,
					n_workers=10, 
					micro_step=40, 
					macro_step=40, 
					save_freq=20
					):

		self.score = score
		self.macro_step = macro_step
		self.micro_step = micro_step

		# Function or Integer supported
		if isinstance(n_workers, int) or isinstance(n_workers, float):
			self.n_workers = lambda step: round(n_workers)
		else:
			self.n_workers = n_workers

		self.paramSpec = paramSpec
		self.workers = []
		self.save_freq = save_freq
		self.save_counter = save_freq
		self.SubjectClass = SubjectClass
		self.output_dir = output_dir
		
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
				w = self.SubjectClass.load(i)
				self.workers.append(w)
				tf.logging.info(f"Loaded {w.id} {self.score(w)}")

			except Exception as e:
				print(e)

		tf.logging.info(f"Loaded workers")



	def scale_workers(self, step_index):

		stack = list(self.workers)
		random.shuffle(stack) # Tie-break randomly
		stack = sorted(stack, key=self.score)
		
		n20 = round(len(stack)*0.2)
		bottom20 = stack[:n20]

		delta = self.n_workers(step_index) - len(self.workers)

		if delta != 0:
			tf.logging.info(f"Resizing worker pool by {delta}")

		if delta < 0:
			self.workers = self.workers[min(-delta, len(self.workers)):]

		elif delta > 0:	
			for i in range(delta):
				additional = self.SubjectClass()
				
				additional.params = {
						k: v() for k, v in self.paramSpec.items()
				}
				additional.count = random.randint(0,round(self.macro_step*0.2))

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
				k:v.mutate() for k, v in params.items()
		}
	
	def ready(self, worker):
		return worker.count > self.macro_step * self.micro_step
	
	def printStatus(self, epoch):

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


				print(f'{{"metric": "{key}_max", "value": {best} }}')
				print(f'{{"metric": "{key}_min", "value": {worst} }}')

		self.hyperplot.add_result(epoch, len(self.workers), "n_workers")

		best_worker = max(self.workers, key=self.score)

		for key, val in best_worker.params.items():
			if isinstance(val.value, int) or isinstance(val.value, float):
				self.hyperplot.add_result(epoch, val.value, f"{key}_best")



	def paramsEqual(self, p1, p2):
		for k, v in p1.items():
			if v != p2[k]:
				return False
		return True
	

	def step(self, step_index):
		for i in self.workers:
			i.step(self.micro_step)
			i.eval()
			tf.logging.info(f"{i.id} eval {self.score(i)}")
			
			if self.ready(i):
				i.resetCount()
				
				params = i.params
				params2 = self.exploit(i)
				
				if not self.paramsEqual(params, params2):
					tf.logging.info(f"Replace with exploit-explore {i.id}")
					i.params = self.explore(params2)
					i.eval()

		self.save_counter -= 1;
		if self.save_counter <= 0:
			self.save()
			self.save_counter = self.save_freq
					
			
		
	def run(self, epochs=1000):
		for i in range(epochs):
			self.scale_workers(i)
			self.step(i)
			self.printStatus(i)
			
