import networkx as nx
import pandas as pd
import numpy as np

from  scipy.sparse import diags, csr_matrix
'''
Calculate the various graph kernels for a graph
'''

__all__=['matrix_power_kernel','matrix_power_kernel_graph']

def matrix_power_kernel_graph(g,method='von_neumann',weight=None,nodes_from=[],nodes_to=None,**kwargs):
	n2idx = {n:i for i,n in enumerate(g.nodes())}
	idx_from = [n2idx[n] for n in nodes_from]
	idx_to = [n2idx[n] for n in nodes_to] if nodes_to else None

	return matrix_power_kernel(nx.adjacency_matrix(g,weight=weight),method,idx_from,idx_to,**kwargs)

def matrix_power_kernel(A, method='von_neumann', idx_from=[], idx_to=None, **kwargs):

	U = None
	
	if method in ['laplace', 'exp_laplace','root_page_rank']:
		import scipy.sparse
		diag = np.array(A.sum(axis=1).flatten().tolist()[0])
		D = diags(diag,0)
		with np.errstate(divide='ignore'):
			inv_diag = 1 /diag
			inv_diag[inv_diag == np.inf] = 0
		D_inv = diags(inv_diag, 0)
		L = D - A

	if method == 'von_neumann':		
		U = __von_neumann__(A,idx_from,idx_to, **kwargs)
	elif method == 'exponential':
		U = __exponential__(A,idx_from,idx_to, **kwargs)
	elif method == 'laplace':
		U = __laplace__(L,idx_from,idx_to, **kwargs)
	elif method == 'exp_laplace':
		U = __exp_laplace__(L,idx_from,idx_to, **kwargs)
	elif method == 'root_page_rank':		
		T = D_inv*L
		U = __root_page_rank(T, idx_from, idx_to, **kwargs)
	else:
		raise Exception('Unknown method')

	return U.T

def __von_neumann__(A, idx_from, idx_to, path_length=12, alpha = 1.0e-3):
	'''
	A: adjacency_matrix
	idx_from,idx_to: lists of indices
	'''
	n_col = len(idx_from)
	
	data_I = np.ones(n_col)
	row_I = idx_from
	col_I = np.arange(n_col)
	U_I = csr_matrix((data_I,(row_I,col_I)), shape=(A.shape[0],n_col))

	U = U_I

	for i in range(1, path_length+1):
		U = A.dot(U)*alpha + U_I

	if idx_to:
		U = U[idx_to,:]
	return U

def __exponential__(A,idx_from,idx_to, path_length=12,alpha = 1.0e-3):
	'''
	A: adjacency_matrix
	idx_from,idx_to: lists of indices
	'''
	n_col = len(idx_from)
	
	data_I = np.ones(n_col)
	row_I = idx_from
	col_I = np.arange(n_col)
	U_I = csr_matrix((data_I,(row_I,col_I)), shape=(A.shape[0],n_col))

	U = U_I

	for i in range(path_length,0,-1):
		U = A.dot(U)*alpha/i + U_I

	if idx_to:
		U = U[idx_to,:]
	return U

def __laplace__(L,idx_from, idx_to, path_length=12, alpha=1.0e-3):

	L = -L
	n_col = len(idx_from)
	
	data_I = np.ones(n_col)
	row_I = idx_from
	col_I = np.arange(n_col)
	U_I = csr_matrix((data_I,(row_I,col_I)), shape=(L.shape[0],n_col))

	U = U_I

	for i in range(path_length,0,-1):
		U = L.dot(U)*alpha/i + U_I

	if idx_to:
		U = U[idx_to,:]
	return U

def __exp_laplace__(L,idx_from, idx_to, path_length=12, alpha=1.0e-3):

	#L = -L
	n_col = len(idx_from)
	
	data_I = np.ones(n_col)
	row_I = idx_from
	col_I = np.arange(n_col)
	U_I = csr_matrix((data_I,(row_I,col_I)), shape=(L.shape[0],n_col))

	U = U_I

	for i in range(path_length,0,-1):
		U = L.dot(U)*alpha/i + U_I

	if idx_to:
		U = U[idx_to,:]
	return U

def __root_page_rank(T, idx_from, idx_to, path_length=12, alpha=1.0e-3):

	U = T[:, idx_from]
	for i in range(path_length):
		U = U + (T.dot(U))*alpha
	
	if idx_to:
		U = U[idx_to,:]
	
	U = U*(1 - alpha)
	return U

class MatrixPowerKernel:

	def __init__(self,kernel='von_neumann',lmax=5,even_step=False,**kwargs):
		
		if kernel not in ['von_neumann','exp_diffusion']:
			raise Exception("Unimplemented kernel:",kernel)
		self.kernel = kernel
		self.lmax = lmax		
		self.even_step = even_step
		
		for k in kwargs:
			self.__setattr__(k,kwargs[k])
		
	def fit(self,A,idx_from=None,idx_to=None):
		'''
		Compute matrix power-based graph kernel.
		Params:
		A: adjacency_matrix
		idx_from: If None ==> Compute centrality
		'''

		M = self.__affinity_matrix__(A)
		
		if idx_from:
			n_col = len(idx_from)	
			data_I = np.ones(n_col)
			row_I = idx_from
			col_I = np.arange(n_col)
			I_Src2All = csr_matrix((data_I,(row_I,col_I)), shape=(M.shape[0],n_col))
			Src2All = I_Src2All.todense()
		else:
			I_Src2All = np.ones(M.shape[0])
			Src2All = I_Src2All
			
		for path_length in range(self.lmax):
			damping_arg = self.__damping_arg__(path_length)
			if self.even_step:
				Src2All = damping_arg*M.dot(M.dot(Src2All)) + I_Src2All				
			else:
				Src2All = damping_arg*M.dot(Src2All) + I_Src2All

		if idx_from and idx_to:
			Src2All = Src2All[idx_to,:]
		elif idx_to:#centrality Src2All is just a vector
			Src2All = Src2All[idx_to]

		return np.asarray(Src2All)

	def fit_with_g(self,g,src=None,dst=None,weight=None):

		A = nx.adjacency_matrix(g,weight=weight)
		n2idx = {n:i for i,n in enumerate(g.nodes())}
		idx_from = [n2idx[n] for n in src] if src else None
		idx_to = [n2idx[n] for n in dst] if dst else None

		return self.fit(A,idx_from,idx_to)

	def __damping_arg__(self, path_length):		
		if self.kernel == 'von_neumann':
			return self.alpha
		elif self.kernel == 'exp_diffusion':
			return self.apha/path_length
	
	def __affinity_matrix__(self,A):
		if self.kernel == 'von_neumann':
			return A
		elif self.kernel == 'exp_diffusion':
			return A

	@classmethod
	def __I_Src2All__(M,idx_from):
		n_col = len(idx_from)	
		data_I = np.ones(n_col)
		row_I = idx_from
		col_I = np.arange(n_col)
		return csr_matrix((data_I,(row_I,col_I)), shape=(M.shape[0],n_col))

