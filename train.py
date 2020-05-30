import datasets
import models

class Train():
	def __init__(self, args):
		self.dataset = datasets.__dict__[args.dataset]()
		self.model = args.model

		self.batch_size = args.batch_size
		self.lr = args.lr
		self.epochs = args.epochs

	def train(self):
		pass

	def validate(self):
		pass

	def execute(self):
		# coarse-training
		for epoch in range(self.epochs):
			self.train(); self.validate();
		# fine-training
		for epoch in range(self.epochs):
			self.train(); self.validate();