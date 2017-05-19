import csv
import pprint as pp
import networkx as nx
import itertools as it
import math
import scipy.sparse
import random


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
    graph = graph_users_items
    g = nx.Graph()
    # Your code here ;)
    users = "vettore users"
    items = "vettore items"
    N = len(users)
    M = len(items)
    for u in range(N):
        for i in range(M):
            j = i + 1
            while (j < M):
                if ():
                    g.add_edge(items[i], items[j], weight=1)
                    weight = weight + 1
                j = j + 1

    return g

gr = create_graph_set_of_users_set_of_items("u_data_homework_format.txt")
print(gr)

#  create_item_item_graph(gr)