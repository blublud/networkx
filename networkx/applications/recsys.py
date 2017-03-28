from sklearn.utils import resample
from collections import OrderedDict
from networkx.algorithms.bipartite import biadjacency_matrix
from numpy import argsort,mean

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

	def recommend(self,n_top_item=0,test_users=None):
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
		
		item_testuser = B.dot(uu_similarity) #dense matrix, shape:(all_item,test_users)
		
		recommended = {}
		
		for i,u in enumerate(test_users):
			similarity = item_testuser[:,i]
			trained_items = B[:,self.u2idx[u]].nonzero()[0]
			similarity[trained_items]=0
			recommended[u] = [self.itemidx2name[i] for i in argsort(similarity)[::-1]]

		return recommended

	def evaluate(self,recommendations):
		'''
		recommendations: dict{u_id:[item1,item2]}
		return score
		'''
		return mean([accuracy(recommended,self.ground_user_items[u]) 
						for u,recommended in recommendations.items()
					])

def accuracy(guessed,truth):
	'''
	guessed,truth: array of item_ids
	'''	
	return len(set.intersection(set(guessed),set(truth)))/min(len(truth),len(guessed))
