import csv
import pprint as pp
import networkx as nx
import itertools as it
import math
import scipy.sparse
import random
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

def pagerank(M, N, nodelist, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, dangling=None):
	if N == 0:
		return {}
	S = scipy.array(M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
	M = Q * M
	
	# initial vector
	x = scipy.repeat(1.0 / N, N)
	
	# Personalization vector
	if personalization is None:
		p = scipy.repeat(1.0 / N, N)
	else:
		missing = set(nodelist) - set(personalization)
		if missing:
			#raise NetworkXError('Personalization vector dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print ('Error: personalization vector dictionary must have a value for every node')
			print
			exit(-1)
		p = scipy.array([personalization[n] for n in nodelist], dtype=float)
		#p = p / p.sum()
		sum_of_all_components = p.sum()
		if sum_of_all_components > 1.001 or sum_of_all_components < 0.999:
			print
			print ("Error: the personalization vector does not represent a probability distribution :(")
			print
			exit(-1)
	
	# Dangling nodes
	if dangling is None:
		dangling_weights = p
	else:
		missing = set(nodelist) - set(dangling)
		if missing:
			#raise NetworkXError('Dangling node dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print ('Error: dangling node dictionary must have a value for every node.')
			print
			exit(-1)
		# Convert the dangling dictionary into an array in nodelist order
		dangling_weights = scipy.array([dangling[n] for n in nodelist], dtype=float)
		dangling_weights /= dangling_weights.sum()
	is_dangling = scipy.where(S == 0)[0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
		# check convergence, l1 norm
		err = scipy.absolute(x - xlast).sum()
		if err < N * tol:
			return dict(zip(nodelist, map(float, x)))
	#raise NetworkXError('power iteration failed to converge in %d iterations.' % max_iter)
	print
	print ('Error: power iteration failed to converge in '+str(max_iter)+' iterations.')
	print
	exit(-1)




def create_graph_set_of_users_set_of_items(user_item_ranking_file):
	graph_users_items = {}
	all_users_id = set()
	all_items_id = set()
	g = nx.DiGraph()
	input_file = open(user_item_ranking_file, 'r')
	input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
	for line in input_file_csv_reader:
		user_id = int(line[0])
		item_id = int(line[1])
		rating = int(line[2])
		g.add_edge(user_id, item_id, weight=rating)
		all_users_id.add(user_id)
		all_items_id.add(item_id)
	input_file.close()
	graph_users_items['graph'] = g
	graph_users_items['users'] = all_users_id
	graph_users_items['items'] = all_items_id
	return graph_users_items
	

def create_item_item_graph(graph_users_items):
    #graph = graph_users_items
    g = nx.Graph()
    # Your code here ;)
    oldGraph = graph_users_items['graph']
    users = graph_users_items['users']
    items = graph_users_items['items']
    N = len(users)
    M = len(items)
    for u in oldGraph.nodes():
        for i in range(N):###################################################################
            j = i + 1
            while (j < M):
                if (nx.items.neighbours(i) == nx.items.neighbours(j)):
                    g.add_edge(items[i], items[j], weight=1)
                j = j + 1

    return g




def create_preference_vector_for_teleporting(user_id, graph_users_items):
    preference_vector = {}
    g = graph_users_items['graph']

    item = graph_users_items['items']
    u = g.neighbors(user_id)
    tot = 0

    for i in item:
        if (i in u):
            value = g[user_id][i]['weight']
            tot = tot + value
        else:
            value = 0
        name = "item_" + str(i)
        preference_vector[name] = value
    preference_vector = {k: v/tot for k, v in preference_vector.items()}
    return preference_vector




def create_ranked_list_of_recommended_items(page_rank_vector_of_items, user_id, training_graph_users_items):
	# This is a list of 'item_id' sorted in descending order of score.
	sorted_list_of_recommended_items = []
	# You can obtain this list from a list of [item, score] couples sorted in descending order of score.
	
	# Your code here ;)
	
	
	return sorted_list_of_recommended_items




def discounted_cumulative_gain(user_id, sorted_list_of_recommended_items, test_graph_users_items):
	dcg = 0.
	# Your code here ;)
	
	
	return dcg
	



def maximum_discounted_cumulative_gain(user_id, test_graph_users_items):
	dcg = 0.
	# Your code here ;)
	
	
	return dcg


def main():
    graph_users_items = {}
    B = nx.Graph()
    all_users_id = set()
    all_items_id = set()
    B.add_nodes_from([1, 2, 3, 4], bipartite=0)  # Add the node attribute "bipartite"
    B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
    B.add_weighted_edges_from((
        [(1, 'c', 3), (2, 'a', 5), (1, 'a', 2), (3, 'a', 1), (1, 'b', 5), (2, 'b',3), (2, 'c',4), (3, 'c',4), (3, 'a',2), (4, 'a',5), (4, 'c',1)]))
    plt.clf()
    nx.draw_networkx(B, pos=nx.spring_layout(B))

    all_users_id.add(1)
    all_users_id.add(2)
    all_users_id.add(3)
    all_users_id.add(4)
    all_items_id.add('a')
    all_items_id.add('b')
    all_items_id.add('c')

    graph_users_items['graph'] = B
    graph_users_items['users'] = all_users_id
    graph_users_items['items'] = all_items_id

    plt.show(block=True)

    #create_item_item_graph(graph_users_items)

    t = create_preference_vector_for_teleporting(4, graph_users_items)


if __name__ == "__main__":
    main()










