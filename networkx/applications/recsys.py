from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from networkx.algorithms.bipartite import biadjacency_matrix
from numpy import array,arange,argsort,mean,ones,zeros,asarray
from scipy.sparse import csr_matrix,csc_matrix

class NeighborRecSys:
	def __init__(self,g,p_test_item,p_test_user=None,test_users=None,is_user=None):
		
		self.g_train = g.copy()
		g=self.g_train

		if is_user:
			self.is_user = is_user
		else:
			self.is_user = lambda n:n.startswith('u')
		
		u_nodes = [n for n in g if self.is_user(n)]
		self.u2idx = OrderedDict([(n,i) for i,n in enumerate(u_nodes)])
		self.itemidx2name = u_nodes = [n for n in g if not self.is_user(n)]

		if p_test_user:
			u_names = [u for u in self.u2idx.keys()]
			self.test_users = resample(u_names,n_samples=len(u_names)*p_test_user,replace=False)
		elif test_users:
			self.test_users = test_users
		else:
			raise Exception("Must set p_test_user of test_users explicitly")

		self.ground_user_items = {}
		for u in self.test_users:
			items = [item for item in g[u].keys()]
			ground_items = resample(items,n_samples=int(len(items)*p_test_item),replace=False)
			self.ground_user_items[u] = ground_items
			g.remove_edges_from([(u,i) for i in ground_items])


	def fit(self,uu_similarity):
		'''
		uu_similarity: matrix all_users-by-test_users
		'''
		assert 	uu_similarity.shape[0] == len(self.u2idx) and \
				uu_similarity.shape[1] == len(self.test_users)

		self.uu_similarity = uu_similarity

	def recommend(self,n_top_user=0,test_users=None):
		'''
		return dict: {uid:[item_ids]}
		'''
		if not test_users:
			test_users = self.test_users
		else:
			test_users = [u for u in test_users if u in self.test_users]

		test_uidx = [i for i,u in enumerate(self.test_users) if u in test_users]

		uu_similarity = self.uu_similarity[:,test_uidx]
		B = biadjacency_matrix(self.g_train,row_order=self.itemidx2name,column_order=self.u2idx.keys())

		if n_top_user > 0:
			sim_user_idx = argsort(uu_similarity,axis=0)[-n_top_user:,:] #shape n_sim_user-by-test_users			
			sim_user_idx = sim_user_idx.flatten(order='F')			
			idx_ptr = arange(0,len(sim_user_idx)+1,n_top_user)
			data = (ones(sim_user_idx.shape),sim_user_idx,idx_ptr)
			select_similar_users = csc_matrix(data,shape=uu_similarity.shape) #binary matrix: each column c has top similar to user c
			item_testuser = asarray(B.dot(select_similar_users.multiply(uu_similarity)))

		else:
			item_testuser = B.dot(uu_similarity) #dense matrix, shape:(all_item,test_users)
		
		recommended = {}
		
		for u_idx,u in enumerate(test_users):
			similarity = item_testuser[:,u_idx]
			trained_items = B[:,self.u2idx[u]].nonzero()[0]
			similarity[trained_items]=0
			#print(argsort(similarity)[::-1])#debug
			recommended[u] = [self.itemidx2name[i] for i in argsort(similarity)[::-1]]

		return recommended

	def evaluate_recommendation(self,recommendations):
		'''
		recommendations: dict{u_id:[item1,item2]}
		return dict{u_id:[prediction_result_1,prediction_result_2]}
		'''
		return	{u:array([r in self.ground_user_items[u] for r in recommended])
					for u,recommended in recommendations.items()}, \
				{u:len(self.ground_user_items[u]) for u in recommendations.keys()}

def accuracy(recommend_result_ranked,n_true_labels):
	'''
	recommend_result_ranked: array of binary flags (RANKED)
	'''
	return sum(recommend_result_ranked)/min(len(recommend_result_ranked),n_true_labels)

def avg_precision_ranked(recommend_result_ranked):
	'''
	Compute average_precision_score for a RANKED recommendation.
	For more details, read sklearn.metrics.average_precision_score
	'''
	precision = [float(i)/(pos+1) for i,pos in enumerate(recommend_result_ranked.nonzero()[0])]
	
	return mean(precision) if len(precision) else .0
