
import numpy as np
import random
import math
from collections import defaultdict
from shapely.geometry import Polygon, Point

# ==================================== HELPER FUNCs =======================================

def finite_polygons(voronoi, diameter):

    '''
    helper function for bounded_polygons().

    Parameter:  
    ----------
    voronoi: Voronoi object
    diameter: float
    
    Return:
    ----------
    Polygon object, generator of finite voronoi regions in the same order of the input points. 
    The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    Reference code: 
    https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region

    '''

    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def finite_noisy_polygons(G, voronoi, diameter, noise):

    '''
    helper_func A for bounded_noisy_polygons(). 
    Return polygons with a fixed length Noise vector. 
    '''

    centroid = voronoi.points.mean(axis=0)
    boundary_polygon = Polygon(np.array(G.graph['boundary']))

    noisy_verticies = np.zeros(voronoi.vertices.shape) 

    for i in range(voronoi.vertices.shape[0]):
        if boundary_polygon.contains(Point(voronoi.vertices[i])):
            flag = True
            while flag:

                ### new version: add fixed size noise with random direction ---
                v = np.random.uniform(low = -1, high = 1, size = 2)
                v_hat = v / np.linalg.norm(v) * noise
                ver = voronoi.vertices[i] + v_hat
                ### -----------------------------------------------------------

                ### old version: add random noise from N(0, noise) ------------
                #ver = voronoi.vertices[i] + np.random.normal(0, noise, size = 2)
                ### -----------------------------------------------------------


                if boundary_polygon.contains(Point(ver)):
                    noisy_verticies[i] = ver
                    flag = False
        else:
            ### new version: add fixed size noise with random direction --------
            v = np.random.uniform(low = -1, high = 1, size = 2)
            v_hat = v / np.linalg.norm(v) * noise
            noisy_verticies[i] = voronoi.vertices[i] + v_hat
            ### ----------------------------------------------------------------

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(noisy_verticies[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = noisy_verticies[region[inf + 1:] + region[:inf]]
        extra_edge = [noisy_verticies[j] + dir_j * length,
                      noisy_verticies[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def sorted_polygon(poly):
    '''
    helper_func B for bounded_noisy_polygons(). 
    The polygon returned form the noisy finite polygons might not be sorted 
    (dots changing sequence locally which will return an error for intersection method)
    '''
    old_face = list(poly.exterior.coords)
    origin = list(poly.centroid.coords)[0]
    refvec = [0, 1]
    def clockwiseangle_and_distance(point):
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    sorted_face = sorted(old_face, key=clockwiseangle_and_distance)
    return Polygon(sorted_face)

def get_random_point_in(poly):

    'helper func for hybrid_seeds()'

    min_x, min_y, max_x, max_y = poly.bounds
    while True:
        p = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if poly.contains(p):
            return p # Point object

# =========================================================================================


def bounded_polygons(G, vor):
    '''
    Parameter:  
    ----------
    G: nx graph object
    vor: Voronoi object
    
    Return:
    ----------
    bounded_regions: list of faces (verties)
    '''
    bounded_regions = []

    # range of value (maximum - minimum); (diameter of the axises x and y)
    diameter = np.linalg.norm(np.array(G.graph['boundary']).ptp(axis=0))

    # boundary must contain all the points:
    boundary_polygon = Polygon(np.array(G.graph['boundary'])).buffer(0)
    
    for p in finite_polygons(vor, diameter):
        
        # periphery polygon intersecting with the bounded area in the blade:

        # when boundary not even, in the case of random dots in the face, 
        # this intersection can return a multi polygon,
        # therefore we filter them out:
        try:
            bounded_geom = p.intersection(boundary_polygon)
            bounded_regions.append(list(bounded_geom.exterior.coords))

        except AttributeError:            
            bounded_regions.append([])
            print('One multipolygon generated...')

    return bounded_regions


def bounded_noisy_polygons(G, vor, noise = 2):
    '''
    Parameter:  
    ----------
    G: nx graph object
    vor: Voronoi object
    
    Return:
    ----------
    bounded_regions: a list of noisy Voronoi Polygons (bounded by the leaf blade).
    '''
    bounded_regions = []

    diameter = np.linalg.norm(np.array(G.graph['boundary']).ptp(axis=0))
    
    boundary_polygon = Polygon(np.array(G.graph['boundary']))
    
    for p in finite_noisy_polygons(G, vor, diameter, noise):
        sorted_p = sorted_polygon(p)
        # big periphery polygon intersecting with the bounded area in the blade:
        bounded_regions.append(list(sorted_p.intersection(boundary_polygon).exterior.coords))

    return bounded_regions


def overlap_test(G, seeds, bounded_regions, type = 'dot'):
    '''
    Parameter:  
    ----------
    G: nx graph object
    seeds: list of coordinates of dots
    bounded_regions: list of faces (list of vertix coordinates)

    Return:
    ----------
    J_list: list of J_index (intersected area/union area)
    '''

    # find the indices of the passed points in the list seed:
    passed_index = []
    for passed_point in G.graph['dots_passed']:
        passed_index.append(seeds.index(passed_point))    


    L = len(G.graph['dots_passed'])
    shared_area_list = np.zeros(L)
    union_area_list = np.zeros(L)

    for i in range(L):
        # if this is not a multi polygon (that we ignore using an empty list[]):
        if bounded_regions[passed_index[i]]:
            shared_shape = Polygon(G.graph['faces_passed'][i]).buffer(0).intersection(Polygon(bounded_regions[passed_index[i]]))
            shared_area_list[i] = shared_shape.area

            union_shape = Polygon(G.graph['faces_passed'][i]).buffer(0).union(Polygon(bounded_regions[passed_index[i]]))
            union_area_list[i] = union_shape.area

    J_list = shared_area_list/union_area_list

    # added 05/13/22: we want to save the differences and plot them on original graphs.
    difference_geom = []

    for i in range(L):
        # if this is not a multi polygon (that we ignore using an empty list[]):
        if bounded_regions[passed_index[i]]:

            diff = Polygon(G.graph['faces_passed'][i]).buffer(0).symmetric_difference(Polygon(bounded_regions[passed_index[i]]))
            difference_geom.append(diff)

    G.graph[f'diff_geom_{type}'] = difference_geom


    return J_list

    
def overlap_noisy_vertices(G, seeds, bounded_regions, noisy_regions):
    '''
    Parameter:  
    ----------
    G: nx graph object
    seeds: list of coordinates of dots
    bounded_regions: list of faces (list of vertix coordinates)
    noisy_regions: list of noisy Voronoi polygons

    Return:
    ----------
    J_list: list of J_index (intersected area/union area)
    '''
    # find the indices of the passed points in the list seed:
    passed_index = []
    for passed_point in G.graph['dots_passed']:
        passed_index.append(seeds.index(passed_point))

    L = len(G.graph['dots_passed'])
    shared_area_list = np.zeros(L)
    union_area_list = np.zeros(L)

    for i in range(L):
        shared_area_list[i] = Polygon(noisy_regions[passed_index[i]]).intersection(Polygon(bounded_regions[passed_index[i]])).area
        union_area_list[i] = Polygon(noisy_regions[passed_index[i]]).union(Polygon(bounded_regions[passed_index[i]])).area

    J_list = shared_area_list/union_area_list
    return J_list



def hybrid_seeds(G):

    '''
    Parameter:  
    ----------
    G: nx graph object

    Return:
    ----------
    centroid_seeds/midpoint_seeds/random_seeds: list of point coordinates, same length as seeds;
                                                replacing the single dots with reference points,
                                                hybrid with the multi-hydathodes.
    '''

    centroid_seeds = []
    midpoint_seeds = []
    random_seeds = []


    for n in G.nodes:
        # if it is not the single dot
        if  G.nodes[n]['type'] == 'dot':
            # copy locaion of of the dot to twoseeding list:
            centroid_seeds.append(n)
            midpoint_seeds.append(n)
            random_seeds.append(n)

        # if is the single dot:
        elif G.nodes[n]['type'] == 'single_dot':
            index = G.graph['dots_passed'].index(n)
            # apend other dots associated with the face the dot is in:
            poly = Polygon(G.graph['faces_passed'][index])
            random_p = get_random_point_in(poly)
            centroid_seeds.append(list(poly.centroid.coords)[0])
            midpoint_seeds.append(list(Polygon(G.graph['faces_passed'][index]).representative_point().coords)[0])
            random_seeds.append(list(random_p.coords)[0])

            
    return centroid_seeds, midpoint_seeds, random_seeds

