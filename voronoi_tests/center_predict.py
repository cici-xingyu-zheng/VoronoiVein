import voronoi_local

import math
import random 
import numpy as np

import networkx as nx

def slope_intercept(shared_edges_mat, adj_faces_mat, *G, test_type = 'random'):

    '''
    Parameter:  
    ----------
    shared_edges_mat: NxN matrix, at entry (i, j) it stores the shared edges of each adjacent polygon pairs i and j.
    adj_faces_mat: boolean NxN matrix, a_ij = 1 if i an j are adjacent
    *G: networkx graph object
    test_type: 'random' or 'best'
    
    Return:
    ----------
    s_list: dictionary storing slopes of all testable pairs, keyed by the indexes of the two flanking polygons
    b_list: dictionary storing intersepts of all testable pairs, keyed by the indexes of the two flanking polygons
    '''

    # with keys being pairs (i,j), values being the slope/intercept
    s_list = {}
    b_list = {} 

    M = len(shared_edges_mat)

    # for each pair of faces:
    for i in range(M):
        for j in range(i+1, M):
            # if they are adjacent:
            if adj_faces_mat[i][j]:
                # Case 1: if more than one edge is shared:
                if len(shared_edges_mat[i][j]) > 1:   
                    
                    # using the best edge:
                    if test_type == 'best':
                        
                        dot_list = G.graph['dots_passed']

                        # gather their test results:
                        rst_list = []
                        for n in range(len(shared_edges_mat[i][j])):
                            
                            vein_node1 = shared_edges_mat[i][j][n][0]
                            vein_node2 = shared_edges_mat[i][j][n][1]
                            dot1 = dot_list[i]
                            dot2 = dot_list[j]

                            rst = voronoi_local.error_calculate(dot1, dot2, vein_node1, vein_node2)
                            rst_list.append(rst)

                        # pick the edge that has the smallest error: 
                        error_list = [rst_list[k][0] for k in range(len(rst_list))]
                        min_index = error_list.index(min(error_list))
                        
                        s, b = voronoi_local.slope_intercept(shared_edges_mat[i][j][min_index][0], shared_edges_mat[i][j][min_index][1])

                    # using an random edge:
                    if test_type == 'random':
                        rand_index = random.randint(0, len(shared_edges_mat[i][j]) -1)
                        s, b = voronoi_local.slope_intercept(shared_edges_mat[i][j][rand_index][0], shared_edges_mat[i][j][rand_index][1])

                        s_list[(i, j)] = s
                        b_list[(i, j)] = b



                # Case 2: only one edge is shared:
                else:
                    vein_node1 = np.array(shared_edges_mat[i][j][0][0])        
                    vein_node2 = np.array(shared_edges_mat[i][j][0][1])
                    
                    s, b = voronoi_local.slope_intercept(vein_node1, vein_node2)

                    s_list[(i, j)] = s
                    b_list[(i, j)] = b

    return s_list, b_list
                    

def solve_lin_system(s_list, b_list, num_dot):

    '''
    Parameter:  
    ----------
    s_list: dictionary storing slopes of all testable pairs, keyed by the indexes of the two flanking polygons
    b_list: dictionary storing intersepts of all testable pairs, keyed by the indexes of the two flanking polygons
    num_dot: the number of single dot hydathodes
    
    Return:
    ----------
    solved_coor: list of coordinates of predicted centers for each single-dot polygon
    '''

    # size of A/B is (k rows, 2n coloumns)
    B = np.zeros([len(s_list), num_dot*2]) 
    zero_vec =  np.zeros(len(s_list))
    for i, (key, _) in enumerate(s_list.items()):

        # match the indices of tru centers:
        B[i][key[0]*2+1] = 1
        B[i][key[1]*2+1] = -1

        if abs(s_list[key]) < 0.0001 and s_list[key] >= 0:
            B[i][key[0]*2] = 10000
            B[i][key[1]*2] = -10000

        elif abs(s_list[key]) < 0.0001 and s_list[key] < 0:
            B[i][key[0]*2] = -10000
            B[i][key[1]*2] = 10000

        else:
            B[i][key[0]*2] = 1/s_list[key] 
            B[i][key[1]*2] = -1/s_list[key]
            
    
    A = np.zeros([len(s_list), num_dot*2]) 
    b_vec = np.zeros(len(s_list))

    for i, (key, _) in enumerate(s_list.items()):
        A[i][key[0]*2] = s_list[key]
        A[i][key[0]*2+1] = -1
        A[i][key[1]*2] = s_list[key]
        A[i][key[1]*2+1] = -1
        b_vec[i] = -2*b_list[key]

     # full mat by stacking A and B:
    full_M = np.vstack((B, A))

    
    full_v = np.hstack((zero_vec, b_vec))

    # least squre solve: 
    solved_coor = np.linalg.lstsq(full_M, full_v, rcond=None)[0]
    solved_coor = solved_coor.reshape((num_dot, 2))

    return solved_coor


def mean_err(predicted_centers, dots):
    '''
    Parameter:  
    ----------
    predicted_centers: list of predicted center coordinates
    dots: list of hydathode coordinates
    
    
    Return:
    ----------
    mean distance of all predicted locations to hydathodes for input list of hydathodes.
    '''

    return np.array( [math.dist(dots[i], predicted_centers[i]) for i in range(len(dots))]).mean()


def slope_intercept_gcc(G, G_dual, shared_edges_mat):

    'version for slope_intercept() with only the giant connected component in the dual taken into account'

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
                    rand_index = random.randint(0, len(shared_edges_mat[index_1][index_2]) -1)
                    s, b = voronoi_local.slope_intercept(shared_edges_mat[index_1][index_2][rand_index][0], shared_edges_mat[index_1][index_2][rand_index][1])

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
    
    'version for solve_lin_system() with only the giant connected component in the dual taken into account'

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




