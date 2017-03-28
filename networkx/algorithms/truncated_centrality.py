import networkx as nx
import numpy as np

def truncated_centrality(Adj, method='von_neumann', **kwargs):
	'''
	Calculate centrality for each node in graph.
	Adj: Adjacency matrix of input graph.
	method: name of method. currently support {'von_neumann','exponential'}
	**kwargs: arguments specific to the graph kernel function (path_length, alpha, etc.)
	'''

	if method not in {'von_neumann', 'exponential'}:
		raise Exception("Unsupported kernel:" + method)


	if method == 'von_neumann':
		return centrality_von_neumann(Adj,**kwargs)


	elif method == 'exponential':		
		return centrality_exponential(Adj,**kwargs)

	return U

def centrality_von_neumann(A, path_length = 6, alpha =5e-3):
	v_1 = np.ones(A.shape[0])
	U = v_1
	
	for i in range(1, path_length+1):
		U = A.dot(U)*alpha + v_1
	return U

def centrality_exponential(A, path_length = 6, alpha =5e-3):
	v_1 = np.ones(A.shape[0])
	U = v_1
	
	for i in range(path_length,0,-1):
		U = A.dot(U)*alpha/i + v_1

	return U