from skspatial.objects import Line, Points, LineSegment
import scipy.stats
from skspatial.plotting import plot_3d
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import itertools
from lidarlinefinder import Line3D, Line_KNN


test1 = {
    "variance" : 0.5,
    "lines" : [(([30,0,0], [0,0,0])), ([5, 5, 3], [5,9,8]),
                ([20, -9, -3], [25,-3,-9]), ([15,10,0], [20,35,0]),
                ([-10, 3, 40], [-2,3,25]), ([2, -20, -13], [-13,1,-10])],
    "interval" : 0.1,
    "bounds" : {'x': (-50, 50), 'y': (-50, 50), 'z': (-50, 50)},
    "probability" : 0.975,
    "initial_number_of_lines" : 3
}

test2 = {
    "variance" : 0.5,
    "lines" : [(([4,0,0], [4,0,3.6576])), ([-6, 0, 0], [-6,0,3.6576])],
    "interval" : 0.05,
    "bounds" : {'x': (-50, 50), 'y': (-15, 50), 'z': (-15, 15)},
    "probability" : 0.975,
    "initial_number_of_lines" : 3
}


def generate_ptcld_from_test(test):
    lines = []
    ptclds = []
    for l in test['lines']:
        l_3d = Line3D(l[0], l[1])
        lines.append(l_3d)
        p = l_3d.generate_pts_from_line(test['interval'], test['variance'])
        ptclds.append(np.array(p))
    
    ptcld_comb = np.concatenate(tuple(ptclds))
    return ptcld_comb, lines

def run_line_knn(test_data, iterations = 20):
    ptcld, lines = generate_ptcld_from_test(test1)
    classifier = Line_KNN(ptcld, test_data['variance'], test_data["probability"], test_data["initial_number_of_lines"])
    
    classifier.fit(20)
    
    fig = plt.figure(1) 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=test_data['bounds']['x'][0], right=test_data['bounds']['x'][1]) 
    ax.axes.set_ylim3d(bottom=test_data['bounds']['y'][0], top=test_data['bounds']['y'][1]) 
    ax.axes.set_zlim3d(bottom=test_data['bounds']['z'][0], top=test_data['bounds']['z'][1])

    #plot ground truth lines in red:
    for l in lines:
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='r')

    for l in classifier.lines:
        points = Points(classifier.line_pointclouds[l])
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
        points.plot_3d(ax, c='b', depthshade=False, s=0.01)
    
    if not classifier.unused_points == []:
        unused_points = Points(classifier.unused_points)
        unused_points.plot_3d(ax, c='k',depthshade=False,s=0.01)

    plt.show()

def visualize_test(test):
    ptcld, lines = generate_ptcld_from_test(test)
    fig = plt.figure(1) 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=test['bounds']['x'][0], right=test['bounds']['x'][1]) 
    ax.axes.set_ylim3d(bottom=test['bounds']['y'][0], top=test['bounds']['y'][1]) 
    ax.axes.set_zlim3d(bottom=test['bounds']['z'][0], top=test['bounds']['z'][1])

    #plot ground truth lines in red:
    for l in lines:
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='k')
    
    plt.show()
    


if __name__ == '__main__':
    # run_line_knn(test1, iterations = 20)
    visualize_test(test2)
    




    




