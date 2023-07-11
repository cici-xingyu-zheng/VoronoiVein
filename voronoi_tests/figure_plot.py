import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import shapely 
from scipy.spatial import voronoi_plot_2d, Voronoi

### Module for plotting 

# set color scheme
order = ['hydathode', 'centroid', 'midpoint', 'random']
colors = ['chocolate', sns.color_palette("rocket_r")[3], sns.color_palette("rocket_r")[4], sns.color_palette("rocket_r")[5]]
palette = dict(zip(order, colors))

def quick_plot(G):
    '''
    Basic visualization.
    Quick plotting for spatial graphs with attr "pos" as a length 2 coordinate vector  
    '''
    node_positions = {}
    color_dict = {'vein':'C0', 'dot': 'C7', 'single_dot': 'C1'}
    node_color = []
    
    for node in G.nodes:
        node_positions[node] = node
        node_color.append(color_dict[G.nodes[node]['type']])
    
    _, ax = plt.subplots(figsize=(9, 9/G.graph['ratio']))
   
    nx.draw(G, pos=node_positions, node_size= 20, node_color= node_color, ax = ax) 
    
    plt.tight_layout()
    plt.show()   
        

def plot_testable(G, G_dual):
    '''
    Basic visualization for tested regions.
    Shade the testable polygons, and plot the dual overlay with the shaded polygons; 
    Can be used after creating the dual graph.
    '''

    node_position_G = {}
        
    node_position_dual = {}


    for node in G.nodes:
        node_position_G[node] = node

    edge_style = ['solid' if G.edges[e]['shared'] =='tested_shared' else 'dashed' for e in G.edges]
    edge_col = ['black' if G.edges[e]['shared'] =='tested_shared' else 'C7' for e in G.edges]

    for node in G_dual.nodes:
        node_position_dual[node] = node

    _, ax = plt.subplots(figsize=(10,10/G.graph['ratio']))

    for i in range(len(G.graph['faces_passed'])):
        p = mpl.patches.Polygon(G.graph['faces_passed'][i], facecolor = 'C7', alpha = .1)
        ax.add_patch(p)

    nx.draw_networkx_edges(G, pos=node_position_G, edge_color = edge_col, style = edge_style, ax = ax) 

    nx.draw(G_dual, pos=node_position_dual, node_size= 20,  node_color= 'seagreen', 
            edge_color ='seagreen',  width = 1, ax = ax)
    
    plt.show()   
  

def plot_baseline(G, G_dual, pt_type = 'centroid'):
    '''
    Basic visualization for reference point sets.
    quick plotting for vein graph and point set being tested, 
    with hydathode and refrence point set specified by `pt_type`
    '''
    node_position_G = {}
    
    compared_position_dual = {}
    node_position_dual = {}
                                                                                                        
    for node in G.nodes:
        node_position_G[node] = node
    
    for node in G_dual.nodes:
        compared_position_dual[node] = (G_dual.nodes[node][pt_type][0], G_dual.nodes[node][pt_type][1])
        node_position_dual[node] = node
         
    _, ax = plt.subplots(figsize=(9,9/G.graph['ratio']))
    
    # later might need two colors for two ind of nodes, dk if networkx do it automatically 
    nx.draw_networkx_edges(G, 
            pos=node_position_G, 
            edge_color = 'C7', ax = ax) 
    nx.draw_networkx_nodes(G_dual, 
                           pos = node_position_dual,  
                           node_size = 20,
                           node_color = 'C1')
    nx.draw_networkx_nodes(G_dual, 
                           pos = compared_position_dual,
                           node_size = 20,
                           node_color = palette[pt_type])
    ax.set_title(f"{pt_type} overlay", fontsize = 16)
    plt.tight_layout()
    plt.show()
           


def plot_dist(df, test = 'angle'):
    '''
    Voronoi I visualization.
    Plot error distribution for Voronoi I tests, with test type specified 
    by 'attr' 
    '''
    _, ax = plt.subplots(nrows =2, figsize = (8, 12))
    
    sns.histplot(df, x = f'{test}_diff', hue = 'type', palette = palette,
                 kde = True, ax = ax[0])

    ax[1] = sns.violinplot(x = f'{test}_diff', y = 'type' , 
                            data = df, palette = palette, inner = 'quartile')
    for l in ax[1].lines:
        l.set_linestyle('--')
        l.set_linewidth(1)
        l.set_color('brown')
        l.set_alpha(0.8)
    for l in ax[1].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    
    ax[0].set_title(f'{test} difference distribution', fontsize = 16)

    plt.show()
    

def plot_voronoi(G, vor):

    '''
    Voronoi II visualization.
    plot overlay for vein graph and Voronoi given all hydathodes.
    '''
    color_dict = {'vein':'C0', 'dot': 'C7', 'single_dot': 'C1'}

    node_positions = {}
    node_color = []
    for node in G.nodes:
        node_positions[node] = node
        node_color.append(color_dict[G.nodes[node]['type']])

    _ , ax = plt.subplots(figsize=(9,9/G.graph['ratio']))
    
    nx.draw(G, pos = node_positions, node_size= 20, node_color= node_color, ax = ax) 

    voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                    line_width=2, line_alpha=0.6, point_size=2, ax = ax)
    plt.show()
     



def plot_overlap_diff(G, bounded_regions, seeds, single_dot, type = 'dot'):

    '''
    Voronoi II visualization.
    Plot regions for Voronoi polygons given all hydathodes,
    with only single dotted polygons's difference with vein graph shaded.
    '''
    _, ax = plt.subplots(figsize = (8, 8/G.graph['ratio']))

    node_positions = {}

    for node in G.nodes:
        node_positions[node] = node

    color = palette[type]

    # plot veins:     
    nx.draw_networkx_edges(G, pos=node_positions, edge_color = 'C7', ax = ax, width = 1.5, alpha = .8) 

    # plot voronoi polygons:
    for i in range(len(bounded_regions)):
        ax.plot(np.array(bounded_regions[i])[:,0], np.array(bounded_regions[i])[:,1], alpha = .7, color = 'C7') # change color to grey

    # plot diff patches:
    for geom in G.graph[f'diff_geom_{type}']:  

        if isinstance(geom, shapely.geometry.polygon.Polygon):
            contour = list(geom.exterior.coords)
           
            if len(contour):
                p = mpl.patches.Polygon(contour, facecolor = color, alpha = .3)
                ax.add_patch(p)
            
            if len(list(geom.interiors)):
                interior = list(geom.interiors[0].coords)
                p_in = mpl.patches.Polygon(interior, facecolor = 'white', alpha = 1)
                ax.add_patch(p_in)

        else: # multipolygon.MultiPolygon
            for ploy in geom:
                contour = list(ploy.exterior.coords)
                if len(contour):
                    p = mpl.patches.Polygon(contour, facecolor = color, alpha = .3)
                    ax.add_patch(p)

                    
                if len(list(ploy.interiors)):
                    interior = list(ploy.interiors[0].coords)
                    p_in = mpl.patches.Polygon(interior, facecolor = 'white', alpha = 1)
                    ax.add_patch(p_in)
    
    # plot seeds:
    ax.scatter(np.array(seeds)[:,0], np.array(seeds)[:,1], s = 10, c = 'C7')

    for i in range(len(bounded_regions)):
        if single_dot[i]:
            ax.scatter(seeds[i][0], seeds[i][1], s = 10, c = palette[type])


    plt.show()


def plot_predicted_voronoi(G, predicted_centers):

    '''
    Voronoi III visualization.
    Plot Voronoi polygons given predicicted centers, 
    on top of vein graph for Voronoi III.
    '''

    vor_solved = Voronoi(predicted_centers)

    node_positions = {}
    color_dict = {'vein':'C7', 'dot': 'C7', 'single_dot': 'C1'}
    node_color = []

    for node in G.nodes:
        node_positions[node] = node
        node_color.append(color_dict[G.nodes[node]['type']])

    dot_list = np.array(G.graph['dots_passed'])


    _, ax = plt.subplots(figsize=(9, 9/G.graph['ratio']))


    nx.draw_networkx_edges(G, pos=node_positions, edge_color = 'C7', ax = ax) 

    ax.scatter(dot_list[:,0], dot_list[:,1], s = 25, c = 'C1')

    for i in range(len(G.graph['faces_passed'])):
        p = mpl.patches.Polygon(G.graph['faces_passed'][i], facecolor = 'C7', alpha = .1)
        ax.add_patch(p)

    voronoi_plot_2d(vor_solved, line_colors= 'C0', 
                                        show_vertices=False, 
                                        alpha = .5, 
                                        ax = ax, 
                                        point_size= 10)


    plt.tight_layout()
    plt.show()   






