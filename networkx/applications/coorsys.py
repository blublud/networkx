from sklearn.utils import resample
from numpy import arange,sqrt,fliplr,argsort
from numpy.linalg import norm
from scipy.linalg import eigh
from collections import OrderedDict,Iterable

import numpy as np

class KernelCoorSys:

	def __init__(self,bipart_node_selector= lambda n:n.startswith('u'),truncate=False,normalize=False):
		if bipart_node_selector:
			self.bipart_node_selector = bipart_node_selector
		else:
			self.bipart_node_selector = lambda n:True
		self.truncate = truncate
		self.normalize = normalize

	def select_nodes(self):
		for n in self.g.nodes():
			if self.bipart_node_selector(n):
				yield n 

	def fit(self,g,kernel,landmarks='high-degree',k_lmrk=5,k_dim=5):
		'''
		landmarks: list of node_names, 'high-degree','centrality', 'random'
		'''
		assert 	(landmarks in ('high-degree','random','centrality')) \
			or 	(type(landmarks) is list) 
		assert (0 < k_dim) and (k_dim <= k_lmrk)

		self.g = g
		self.kernel =kernel
		self.k_dim = k_dim
		self.node2idx = OrderedDict([(n,i) for i,n in enumerate(self.select_nodes())])

		if landmarks in ['high-degree','centrality', 'random']:
			self.landmarks = self.__get_landmarks__(landmarks,k_lmrk)
		else:
			self.landmarks = landmarks

		lmrk_idx = [self.node2idx[n] for n in self.landmarks]
		A2L = kernel.fit_with_g(g,src=self.landmarks,dst=self.node2idx.keys())
		if self.truncate:
			# L2L is A2L[lmrk_idx,:]
			#truncate:
			Sig,U = eigh(A2L[lmrk_idx,:])
			min_eigval = min(Sig[Sig>0])
			A2L[lmrk_idx,arange(k_lmrk)] = A2L[lmrk_idx,arange(k_lmrk)] - min_eigval

		L2L = A2L[lmrk_idx,:]
		
		if self.normalize:
			#find eigenpairs of normalized L2L
			L2L = A2L[lmrk_idx,:]
			sqrt_d = sqrt(L2L.diagonal())		
			L2L = (L2L / sqrt_d) / sqrt_d[:,None]
			print(L2L)
			#half-normlize A2L:		
			A2L = A2L / sqrt_d
			
		self.A2L = A2L
		Sig,U = eigh(L2L,eigvals=(k_lmrk-k_dim,k_lmrk-1))
		self.Sig = Sig[::-1]
		self.U = fliplr(U)

		return self

	def __get_landmarks__(self,strategy,k_lmrk):
		'''
		strategy: how to choose landmarks. Must be one of ['high-degree','centrality', 'random']
		'''
		g = self.g
		kernel = self.kernel
		if strategy =='high-degree':			
			node_degree = g.degree(self.select_nodes())
			return [n for n,_ in sorted(node_degree,key=lambda x:x[1])][-k_lmrk:]
		elif strategy == 'random':
			return resample([n for n in self.select_nodes()],n_samples=k_lmrk,replace=False)
		elif strategy == 'centrality':
			nodes = [n for n in self.select_nodes()]
			centrality = kernel.fit_with_g(g,src=None,dst=None)			
			return [n for n,_ in sorted(zip(g.nodes(),centrality), key=lambda x:x[1]) if self.bipart_node_selector(n)][-k_lmrk:]
			
	def __getitem__(self,node):
		if isinstance(node,str):
			return self.__get_coor__([node])
		elif node is None or isinstance(node,Iterable):
			return self.__get_coor__(node)
		else:
			return None

	def __get_coor__(self,nodes):
		if nodes:
			idx = [self.node2idx[n] for n in nodes]
			X2L = self.A2L[idx,:]
		else:
			X2L = self.A2L

		if self.normalize:		
			norms = norm(X2L,axis=1)
			X2L = X2L / norms[:,None]		
		X2L = X2L.dot(self.U[:,:self.k_dim]) / sqrt(self.Sig[:self.k_dim])
		return X2L
