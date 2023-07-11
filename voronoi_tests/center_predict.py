import voronoi_local # local module

import math
import random 
import itertools

import numpy as np
import networkx as nx

### This module performs Voronoi III test.

def mean_err(predicted_centers, dots):
    '''
    Parameter:  
    ----------
    predicted_centers: list of predicted center coordinates
    dots: list of hydathode coordinates
    
    
    Return:
    ----------
    np array of mean distance of all predicted locations to 
    hydathodes for input list of hydathodes.
    '''

    return np.array([math.dist(dots[i], predicted_centers[i]) for i in range(len(dots))]).mean()



def slope_intercept_gcc(G, G_dual, shared_edges_mat, test_type = "best"):

    
    '''
    Calculate the coefficient that will be feed into the linear system.

    Parameter:  
    ----------
    G: networkX object, the leaf graph.
    G_dual: networkX object, the dual graph of hydathode, 
            with edge suggesting adjacency
    shared_edges_mat: NxN matrix, at entry (i, j) it stores the shared edges 
            of each adjacent polygon pairs i and j.
    adj_faces_mat: boolean NxN matrix, a_ij = 1 if i an j are adjacent
    *G: networkx graph object
    test_type: 'random' or 'best'
    
    Return:
    ----------
    s_list: dictionary storing slopes of all testable pairs, 
            keyed by the indexes of the two flanking polygons
    b_list: dictionary storing intersepts of all testable pairs, 
            keyed by the indexes of the two flanking polygons
    g_list: list of hydathode nodes for the gcc of the dual
    g_nodes_index: list of indices of the nodes in g_list when they 
            appear in the `passed` node list.

    '''

    giant_dual =  G_dual.subgraph(max(nx.connected_components(G_dual), key=len))
    g_list = list(giant_dual.nodes())
    passed = G.graph['dots_passed']
    g_nodes_index = {}
    for node in g_list:
        g_nodes_index[node] = passed.index(node)
    s_list = {}
    b_list = {} 

    for i in range(len(g_list)):
        node_1 = g_list[i]
        for j in range(i, len(g_list)):
            node_2 = g_list[j]
            if giant_dual.has_edge(node_1, node_2):
                index_1 = g_nodes_index[node_1] 
                index_2 = g_nodes_index[node_2] 

                # Case 1: if more than one edge is shared:
                if len(shared_edges_mat[index_1][index_2]) > 1:

                    ## random selection
                    if test_type == 'random':

                        rand_index = random.randint(0, len(shared_edges_mat[index_1][index_2]) -1)
                        s, b = voronoi_local.slope_intercept(shared_edges_mat[index_1][index_2][rand_index][0], shared_edges_mat[index_1][index_2][rand_index][1])

                        s_list[(node_1, node_2)] = s
                        b_list[(node_1, node_2)] = b

                    # best local selection
                    if test_type == 'best':

                        dot_list = G.graph['dots_passed']
                        # gather their test results:
                        rst_list = []

                        for n in range(len(shared_edges_mat[index_1][index_2])):
                            
                            vein_node1 = shared_edges_mat[index_1][index_2][n][0]
                            vein_node2 = shared_edges_mat[index_1][index_2][n][1]
                            dot1 = dot_list[i]
                            dot2 = dot_list[j]

                            rst = voronoi_local.error_calculate(dot1, dot2, vein_node1, vein_node2)
                            rst_list.append(rst)

                        # pick the edge that has the smallest error: 
                        error_list = [rst_list[k][1] for k in range(len(rst_list))]
                        min_index = error_list.index(min(error_list))
                        
                        s, b = voronoi_local.slope_intercept(shared_edges_mat[index_1][index_2][min_index][0],
                                                             shared_edges_mat[index_1][index_2][min_index][1])
                    
                        s_list[(node_1, node_2)] = s
                        b_list[(node_1, node_2)] = b


                # Case 2: only one edge is shared:
                else:
                    vein_node1 = np.array(shared_edges_mat[index_1][index_2][0][0])        
                    vein_node2 = np.array(shared_edges_mat[index_1][index_2][0][1])
                    
                    s, b = voronoi_local.slope_intercept(vein_node1, vein_node2)

                    s_list[(node_1, node_2)] = s
                    b_list[(node_1, node_2)] = b
    
    
    return s_list, b_list, g_list, g_nodes_index


def solve_lin_system_gcc(s_list, b_list, g_list):
    
    '''
    Solve for the predicted centers for the gcc of of the dual graph.

    Parameter:  
    ----------
    s_list: dictionary storing slopes of all testable pairs, 
            keyed by the indexes of the two flanking polygons
    b_list: dictionary storing intersepts of all testable pairs, 
            keyed by the indexes of the two flanking polygons
    g_list: list of the hydathode nodes of the gcc.

    Return:
    ----------
    solved_coor: list of coordinates of predicted centers for the gcc of the dual.
    '''   

    B = np.zeros([len(s_list), len(g_list)*2]) 
    zero_vec =  np.zeros(len(s_list))

    for i, (key, s) in enumerate(s_list.items()):
        index_1 = g_list.index(key[0])
        index_2 = g_list.index(key[1])

        B[i][index_1*2+1] = 1
        B[i][index_2*2+1] = -1

        if abs(s) < 0.0001 and s >= 0:
            B[i][index_1*2] = 10000
            B[i][index_2*2] = -10000

        elif abs(s) < 0.0001 and s < 0:
            B[i][index_1*2] = -10000
            B[i][index_2*2] = 10000

        else:
            B[i][index_1*2] = 1/s
            B[i][index_2*2] = -1/s

    A = np.zeros([len(b_list), len(g_list)*2]) 
    b_vec = np.zeros(len(b_list))

    for i, (key, b) in enumerate(b_list.items()):
        index_1 = g_list.index(key[0])
        index_2 = g_list.index(key[1])

        A[i][index_1*2] = s_list[key]
        A[i][index_1*2+1] = -1
        A[i][index_2*2] = s_list[key]
        A[i][index_2*2+1] = -1
        b_vec[i] = -2*b

    # full mat by stacking A and B:
    full_M = np.vstack((B, A))
    full_v = np.hstack((zero_vec, b_vec))

    # least squre solve: 
    solved_coor = np.linalg.lstsq(full_M, full_v, rcond=None)[0]
    solved_coor = solved_coor.reshape((len(g_list), 2))

    return solved_coor


def random_all(G, G_dual, giant_dual, shared_edges_mat, g_list, node_list):

    '''
    Get the best center prediction solution from randomly draw from the mulitple edge cases
    (100*the number of such cases) times. "best" is being judged by the distance to the 
    point set (node_list) being compared. 

    Parameter:  
    ----------
    G: nx Graph
    G_dual: nx Graph, the hydathodes with edge suggesting adjacency 
    giant_dual: nx Graph, the gcc of the dual graph
    shared_edges_mat: adjacency matrix containing tuples of shared edges
    g_list: list of hydathode nodes in the gcc of the dual
    node_list: list of reference point set coordinates that match the ones in the g_list

    Return:
    ----------
    ceiling: float, solution with the smallest error over 100* # multi-edge
    predicted_centers_at_ceiling: list of coordinates of predicted centers with smallest error
    '''   

    mult_num = 0
    passed = G.graph['dots_passed']
    g_nodes_index = {}
    for node in g_list:
        g_nodes_index[node] = passed.index(node)

    for i in range(len(g_list)):
        node_1 = g_list[i]
        for j in range(i, len(g_list)):
            node_2 = g_list[j]
            if giant_dual.has_edge(node_1, node_2):
                index_1 = g_nodes_index[node_1] 
                index_2 = g_nodes_index[node_2] 

                # Case 1: only one edge is shared:
                if len(shared_edges_mat[index_1][index_2]) > 1:
                    mult_num += 1

    ceiling = 1000000
    
    for i in range(100*mult_num):

        s_list, b_list, g_list, g_nodes_index = slope_intercept_gcc(G, G_dual, shared_edges_mat, test_type= 'random')
        predicted_centers = solve_lin_system_gcc(s_list, b_list, g_list)
        
        mean_dist = mean_err(predicted_centers, node_list)
        if mean_dist < ceiling:
            ceiling = mean_dist
            predicted_centers_at_ceiling = predicted_centers
    
    return ceiling, predicted_centers_at_ceiling


