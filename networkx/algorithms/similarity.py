'''
Measure similarity from one node to others
'''
import networkx as nx

__all__=['similarity_unweighted_projection', 'similarity_snp', 'similarity_katz','similarity_jaccard','similarity_adamic']

def similarity_snp(uu_g, from_node, to_nodes=None, p_expose=.001, max_iter=100,tol=1.0e-6,weight='weight'):
    
    q_expose = 1-p_expose
    nnodes=uu_g.number_of_nodes()
    
    snp=dict((n,0) for n in uu_g)
    for nbr in uu_g[from_node]:
        snp[nbr]=(1-q_expose**uu_g[from_node][nbr][weight])

    for i in range(max_iter):        
        prev_snp=snp
        snp=dict.fromkeys(prev_snp,0)
        for n in snp:
            for nbr in uu_g[n]:
                snp[nbr]+=prev_snp[n]*(1-q_expose**uu_g[n][nbr][weight])
    
        err=sum([abs(snp[n]-prev_snp[n]) for n in uu_g])
        #Check for convergence
        if err < tol*nnodes:
            if to_nodes:
                return {n:snp[n] for n in to_nodes}
            else:
                return snp
    
    raise nx.NetworkXError("Cannot converge")

def similarity_katz(uu_g, from_node, to_nodes=None, alpha=.1, max_iter=100,tol=1.0e-6,weight='weight'):
    
    nnodes=uu_g.number_of_nodes()
    x=dict((n,0) for n in uu_g)
    for nbr in uu_g[from_node]:
        x[nbr]=alpha
		
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication y^T = Alpha * x^T A - Beta
        for n in x:
            for nbr in uu_g[n]:
                x[nbr] += xlast[n] * uu_g[n][nbr].get(weight, 1)
        for n in x:
            x[n] = alpha*x[n]

        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*tol:
            if to_nodes:
                return {n:x[n] for n in to_nodes}
            else:
                return x

    raise nx.NetworkXError('Power iteration failed to converge in '
                           '%d iterations.' % max_iter)

def similarity_unweighted_projection(bipart,from_node, to_nodes=None, sim_func=None):
    
    if sim_func is None:
        sim_func=lambda B,u,v: 1 if (set(B[u]) & set(B[v])) else 0
    
    B=bipart
    u=from_node
    unbrs = set(B[u])
    nbrs2 = set((n for nbr in unbrs for n in B[nbr])) - set([u])
    
    sims={}
    
    for v in set(nbrs2)&(set(to_nodes) if to_nodes else set([])):
        sim = sim_func(B,u,v)
        sims[v:sim]
    
    return sims

def similarity_jaccard(bipart, from_node, to_nodes=None):
    
    def jaccar_sim_func(B,u,v):
        unbrs = set(B[u])
        vnbrs = set(B[v])
        return float(len(unbrs & vnbrs)) / len(unbrs | vnbrs)
    
    return similarity_unweighted_projection(bipart,from_node,to_nodes, sim_func=jaccar_sim_func)

def similarity_adamic(bipart, from_node, to_nodes=None):
    def adamic_sim_func(B,u,v):
        return sum([1.0/math.log(len(B[common_nbr])) for common_nbr in (set(B[u].keys()) & set(B[v].keys()))])
    
    return similarity_unweighted_projection(bipart,from_node,to_nodes,sim_func=adamic_sim_func)
