# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from model_basic import BasicModel
# To import modules in utils folder
import os, sys
import numpy as np
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

def var_create(shape, requires_grad=True):
	var = Variable(torch.zeros(shape), requires_grad=requires_grad)
	var.data = torch.from_numpy(np.random.normal(size=shape))
	return var.float()

class GrCNNEncoder(nn.Module):
	def __init__(self, config):
		super(GrCNNEncoder, self).__init__()
		self.config = config
		self.add_model_variables()
		self.cls_projection_layer = nn.Linear(config.hidden_size, config.cls_hidden_size)

	def add_model_variables(self):
		config = self.config
		'''
		self.U_val = var_create(shape=(config.wordvec_size, config.hidden_size))
		self.W_l = var_create(shape=(config.hidden_size, config.hidden_size))
		self.W_r = var_create(shape=(config.hidden_size, config.hidden_size))
		self.W_b = Variable(torch.zeros(config.hidden_size), requires_grad=True)
		self.G_l = var_create(shape=(config.hidden_size, 3))
		self.G_r = var_create(shape=(config.hidden_size, 3))
		self.G_b = Variable(torch.zeros(3), requires_grad=True)
		'''
		self.U_val = nn.Parameter(torch.Tensor(config.wordvec_size, config.hidden_size))
		self.W_l = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
		self.W_r = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
		self.W_b = nn.Parameter(torch.zeros(config.hidden_size))

		self.G_l = nn.Parameter(torch.Tensor(config.hidden_size, 3))
		self.G_r = nn.Parameter(torch.Tensor(config.hidden_size, 3))
		self.G_b = nn.Parameter(torch.zeros(3))
		
		self.weights = [self.U_val, self.W_l, self.W_r, self.G_l, self.G_r]
		self.bias = [self.W_b, self.G_b]

		for w in self.weights:
			w.data.uniform_(-0.1, 0.1)

	def add_embedding(self, pre_trained_embedding=None):
		config = self.config
		if self.config.embedding_random_flag:
			embedding = nn.Embedding(config.vocab_size, config.wordvec_size)
		else:
			embedding = nn.Embedding(config.vocab_size, config.wordvec_size)
			embedding.weight = nn.Parameter(pre_trained_embedding)
		self.embedding = embedding


	def Tree_layer(self, inputs):
		pyramids = []
		inputs = inputs.view(-1, config.wordvec_size)
		current_level = inputs.mm(self.U_val).view(-1, config.num_steps, config.hidden_size)
		pyramids.append(current_level)
		for i in xrange(self.steps-1):
			left_current_level = current_level[:,:self.steps-i-1,:].clone()
			#print(left_current_level.size())
			_left_current_level = left_current_level.view(-1, config.hidden_size)
			#print(_left_current_level.size())
			right_current_level = current_level[:,1:self.steps-i,:].clone()
			_right_current_level = right_current_level.view(-1, config.hidden_size)
			#print(_left_current_level.size())
			_central_current_level = _left_current_level.mm(self.W_l) + \
									 _right_current_level.mm(self.W_r) 
			_central_current_level += self.W_b.expand_as(_central_current_level)
			_central_current_level = F.relu(_central_current_level)

			_current_gate = _left_current_level.mm(self.G_l) + _right_current_level.mm(self.G_r)
			_current_gate += self.G_b.expand_as(_current_gate)
			current_gate = F.softmax(_current_gate)
			
			_left_gate, _central_gate, _right_gate = \
				current_gate[:,0].clone(), current_gate[:,1].clone(), current_gate[:,2].clone()
			left_gate = _left_gate.view(-1,1).repeat(1, config.hidden_size)
			central_gate = _central_gate.view(-1,1).repeat(1, config.hidden_size)
			right_gate = _right_gate.view(-1,1).repeat(1, config.hidden_size)
			#left_gate = _left_gate.view(-1,1)
			#central_gate = _central_gate.view(-1,1)
			#right_gate = _right_gate.view(-1,1)

			_next_level = left_gate.mul(_left_current_level) + \
						 central_gate.mul(_central_current_level) + \
						 right_gate.mul(_right_current_level)

			next_level = _next_level.view(-1,self.steps-i-1,config.hidden_size)
			#print(next_level.size())
			pyramids.append(next_level)
			current_level = next_level
		return pyramids

	def pooling_layer(self, pyramids):
		pooled = []
		for p in pyramids:
			if config.pooling == 'max':
				pooled_hidden, _ = torch.max(p, dim=1)
			if config.pooling == 'average':
				pooled_hidden, _ = torch.mean(p, dim=1)
			pooled.append(pooled_hidden) 

		pooled_tensor = torch.cat(pooled, dim=1)
		return pooled_tensor

	def gating_layer(self, inputs):
		pass

	def cls_layer(self, inputs):
		config = self.config
		assert inputs.size()[-1] == config.hidden_size
		inputs = inputs.view(-1, config.hidden_size)
		cls_hidden = F.relu(self.cls_projection_layer(inputs))
		return cls_hidden

	def forward(self, inputs):
		assert len(inputs.size()) == 2
		self.batch_size, self.steps = list(inputs.size())
		config = self.config
		_inputs = self.embedding(inputs)
		self.pyramids = self.Tree_layer(_inputs)
		self.pooled_tensor = self.pooling_layer(self.pyramids)
		print(self.pooled_tensor.size())
		#self.cls_hidden = F.relu(self.cls_projection_layer(self.pooled_tensor.view(-1,config.hidden_size))).view(-1, self.steps, config.hidden_size)
		self.cls_hidden = self.cls_layer(self.pooled_tensor).view(-1, self.steps, config.cls_hidden_size)
		print(self.cls_hidden.size())
		
class Config(object):
	def __init__(self):
		self.cls_hidden_size = 8
		self.gating_layer_hidden_size = 16
		self.hidden_size = 2
		self.num_steps = 5
		self.num_classes = 3
		self.wordvec_size = 20
		self.embedding_random_flag = True
		self.vocab_size = 100
		self.batch_size = 8
		self.pooling = 'max'

if __name__ == "__main__":
	use_cuda = torch.cuda.is_available()
	config = Config()
	GrCNN = GrCNNEncoder(config)
	GrCNN.add_embedding()
	fake_input = np.random.randint(10, size=(2,5))
	GrCNN(Variable(torch.from_numpy(fake_input)))
	#print(GrCNN.embedding)
	'''def fn(previous_output, current_input):
		return previous_output + current_input

	elems = tf.Variable([1.0, 2.0, 2.0, 2.0])
	elems = tf.identity(elems)
	initializer = tf.constant(0.0)
	out = tf.scan(fn, elems, initializer=initializer)

	with tf.Session() as sess:
	    sess.run(tf.initialize_all_variables())
	    print(sess.run(out))'''