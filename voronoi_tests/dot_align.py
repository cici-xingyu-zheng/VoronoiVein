
import graph_create # local module 

import cv2
import networkx as nx

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

### This module performs the 
### ordinary Procrustes analysis style alignment.
### It's own plotting functions are included.

# ==================================== HELPER FUNCs =======================================


def add_pet_n_tip(G, tip_file, pet_file):
    
    '''
    Helper func for anchored(). adding new node type as tip and pet(iole) to the graph.
    '''

    img = cv2.imread(tip_file, 0)
    _, threshed = cv2.threshold(img, 100, 255,  cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # output contour list for dots:
    contour = cv2.findContours(threshed, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contour) == 1:
        
        x_cor = np.around(np.array([contour[0][i][0][0] for i in range(contour[0].shape[0])]).mean(), 2)
        y_cor = np.around(np.array([contour[0][i][0][1] for i in range(contour[0].shape[0])]).mean(), 2)

    tip = (x_cor, y_cor)
    G.graph['tip'] = tip
    G.add_node(tip, type = 'tip')   


    img = cv2.imread(pet_file, 0)

    _, threshed = cv2.threshold(img, 100, 255,  cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # output contour list for dots:
    contour = cv2.findContours(threshed, cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contour) == 1:
        x_cor = np.around(np.array([contour[0][i][0][0] for i in range(contour[0].shape[0])]).mean(), 2)
        y_cor = np.around(np.array([contour[0][i][0][1] for i in range(contour[0].shape[0])]).mean(), 2)

    pet = (x_cor, y_cor)
    G.graph['pet'] = pet
    G.add_node(pet, type = 'pet') 

def get_vec(G):

    'helper func for change_pos(). Get the unit vector (direction) and the length (scale)'

    vec = (G.graph['pet'][0]- G.graph['tip'][0], G.graph['pet'][1]- G.graph['tip'][1])
    dist = np.linalg.norm(vec)
    unit_vec = vec /dist

    return unit_vec, dist

def get_angle(vec, vec_prime):

    'helper func for change_coor(); get angle between two vectors'

    dot_product = np.dot(vec, vec_prime)
    angle = np.arccos(dot_product)

    return angle

def rotate(origin, point, angle):

    """
    helper_func for correct_pos().
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point


    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    
    return (qx, qy)
# =========================================================================================


def anchored(sample, dot_folder, vein_folder):
    '''
    Parameter:  
    ----------
    sample: string, sample name
    dot_folder: string, path to the hydathode file
    vein_folder: string, path to the vein graph file
    
    Return:
    ----------
    G: nx.graph object, has tip and petole coordinates stored. 
    '''

    print('Anchoring:', sample, '\n')

    G = graph_create.graph_creation(sample, dot_folder, vein_folder)

    tip_file = f'dot_alignment/{sample}_tip.jpg'
    pet_file = f'dot_alignment/{sample}_pet.jpg'
    add_pet_n_tip(G, tip_file, pet_file)

    print('- Addition Step6: finish adding anchor points. \n')
    return G


def create_pos(G):
    '''
    Create pos attribute for nodes; which will be scaled and rotated. 
    Nodes will still be the original coordinates.
    '''

    for node in G.nodes:
        G.nodes[node]['pos'] = node


def correct_pos(G, rotation):
    
    '''
    Parameter:  
    ----------
    G: nx.graph object, anchored graph
    rotation: float, degree of rotation.
    
    Return:
    ----------
    G: nx.graph object, with orientation of the anchored graph corrected by the rotation degree (node.pos)
    The function is used when we want to make the leaf graph align vertical or horizontal. 
    Graph attributes "boundary", "faces_passed" and "dots_passed" will all have corrected positions.
    '''

    for node in G.nodes:
        pos = rotate(G.graph['pet'], G.nodes[node]['pos'], rotation)
        G.nodes[node]['pos'] = pos
    
    G.graph['tip'] =rotate(G.graph['pet'], G.graph['tip'], rotation)

    new_boundary = []

    for bound_node in G.graph['boundary']:
        new_bnode = rotate(G.graph['pet'], bound_node, rotation)
        new_boundary.append(new_bnode)

    G.graph['boundary'] = new_boundary

    # update the location of faces and dots for J_similarity test: 
    new_faces = []
    for face in G.graph['faces_passed']:
        new_face = []
        for node in face:
            new_face.append(rotate(G.graph['pet'], node, rotation))
        new_faces.append(new_face)
    G.graph['faces_passed'] = new_faces

    new_dots = []
    for dot in G.graph['dots_passed']:
        new_dots.append(rotate(G.graph['pet'], dot, rotation))
    G.graph['dot_passed'] = new_dots

    return G


def change_coor(G, G_prime):

    '''
    Parameter:  
    ----------
    G: nx.graph object, anchored graph
    G_prime: nx.graph object, to be stretched to the same orientation and scale as G

    Return:
    ----------
    G_prime: nx.graph object, with 'pos' scaled and rotated to match that of the G graph.
    Graph attributes "boundary", "faces_passed" and "dots_passed" will all have the scaled positions, too.
    '''
    
    norm_vec, dist = get_vec(G)
    norm_vec_prime, dist_prime = get_vec(G_prime)

    angle = get_angle(norm_vec, norm_vec_prime)

    # add a sign to the angle!
    if norm_vec[0]*norm_vec_prime[1] - norm_vec[1]*norm_vec_prime[0] > 0:
        angle = - angle 

    streach = dist/dist_prime

    x_move = G.graph['pet'][0] -G_prime.graph['pet'][0] 
    y_move = G.graph['pet'][1] -G_prime.graph['pet'][1] 

   
    for node in G_prime.nodes:

        # shift:
        new_pos = (G_prime.nodes[node]['pos'][0] + x_move, G_prime.nodes[node]['pos'][1] + y_move)


        # streach:
        new_pos =(G.graph['pet'][0] + (new_pos[0] - G.graph['pet'][0])*streach,
                G.graph['pet'][1] + (new_pos[1] - G.graph['pet'][1])*streach)

        # rotate:
        new_pos = rotate(G.graph['pet'], new_pos, angle)

        
        # assign new pos:
        G_prime.nodes[node]['pos'] = new_pos   
    
    # update the location of faces and dots for J_similarity test:
    new_faces = []
    for face in G_prime.graph['faces_passed']:
        new_face = []
        for node in face:
            new_face.append(G_prime.nodes[node]['pos'])
        new_faces.append(new_face)
    G_prime.graph['faces_passed'] = new_faces

    new_dots = []
    for dot in G_prime.graph['dots_passed']:
        new_dots.append(G_prime.nodes[dot]['pos'])
    G_prime.graph['dot_passed'] = new_dots
    
    new_boundary = []

    for node in G_prime.graph['boundary']:

        new_x = node[0] + x_move
        new_y = node[1] + y_move

        (new_x, new_y) = rotate(G.graph['pet'], (new_x, new_y), angle) 

        new_x = G.graph['pet'][0] + (new_x - G.graph['pet'][0])*streach
        new_y = G.graph['pet'][1] + (new_y - G.graph['pet'][1])*streach

        new_boundary.append((new_x, new_y))

    G_prime.graph['boundary'] = new_boundary

    G_prime.graph['pet'] = G.graph['pet']
    G_prime.graph['tip'] = G.graph['tip']


    return G_prime




# ================================= plotting functions =====================================

def quick_plot(G):
    '''
    quick plotting for spatial graphs with attr "pos" as a length 2 coordinate vector  
    BEFORE we change corr.

    This func is used to diagnoise whether "pet" node and "tip" node are successfully added.
    '''
    node_positions = {}
    color_dict = {'vein':'C0', 'dot': 'C7', 'single_dot': 'C1', 'tip':  'C4','pet':'C5'}
    node_color = []
    
    for node in G.nodes:
        node_positions[node] = node
        node_color.append(color_dict[G.nodes[node]['type']])
    
    _, ax = plt.subplots(figsize=(6, 6/G.graph['ratio']))
   
    nx.draw(G, pos=node_positions, node_size= 30, node_color= node_color, ax = ax) 
    
    plt.tight_layout()
    plt.show()   


# after we do the transformations, we don't plot nodes by their names 
# but give them a mutable position attribute.
def plot_pos(G):
    '''
    quick plotting for spatial graphs with attr "pos" as a length 2 coordinate vector  
    '''
    node_positions = {}
    color_dict = {'vein':'C0', 'dot': 'C7', 'single_dot': 'C1', 'tip':  'C4','pet':'C5'}
    node_color = []
    
    for node in G.nodes:
        node_positions[node] = G.nodes[node]['pos']
        node_color.append(color_dict[G.nodes[node]['type']])
    
    _, ax = plt.subplots(figsize=(6, 6*G.graph['ratio'])) # since we rotate 90 the ratio is the exact opposite.
   
    nx.draw(G, pos=node_positions, node_size= 30, node_color= node_color, ax = ax) 
    
    plt.tight_layout()
    plt.show()


def draw_nodes(G_sub, ax, color = 'deepskyblue'):
    
    'helper func to plot only HYDATHODEs from one sample; used to overlay hydathodes from different samples'

    node_positions = {}

    for node in G_sub.nodes:
        node_positions[node] = G_sub.nodes[node]['pos']
    
    nx.draw(G_sub, pos=node_positions, node_size= 30,  node_color = color,  ax = ax) 


def draw_vein(G_vein, ax, width = 3, alpha = .5):
    
    'helper func to plot only VEINs from one sample; used to overlay veins of different samples'

    vein_positions = {}

    for node in G_vein.nodes:
        vein_positions[node] = G_vein.nodes[node]['pos']
    
    nx.draw_networkx_edges(G_vein, pos = vein_positions, edge_color= 'C7', ax = ax, width = width, alpha = alpha)

# =========================================================================================

