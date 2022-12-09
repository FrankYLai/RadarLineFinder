from skspatial.objects import Line, Points, LineSegment
import scipy.stats
from skspatial.plotting import plot_3d
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from lidarlinefinder import Line3D, KLinesCluster

test1 = {
    "variance" : 0.2,
    "lines" : [(([300,0,0], [300,0,3.6576])), ([-300, 0, 0], [-300,0,3.6576]),
               (([0,300,0], [0,300,3.6576])), ([0, -300, 0], [0,-300,3.6576])],
    "interval" : 0.05,
    "bounds" : {'x': (-330, 330), 'y': (-330, 330), 'z': (-2, 6)},
    "probability" : 0.975,
    "initial_number_of_lines" : 10,
    "noise" : 0.9
}

test2 = {
    "variance" : 0.5,
    "lines" : [(([30,0,0], [0,0,0])),(([-10,30,0], [0,0,0])), ([5, 5, 3], [5,9,8]),
                ([20, -9, -3], [25,-3,-9]), ([15,10,0], [20,35,0]),
                ([-10, 3, 40], [-2,3,25]), ([2, -20, -13], [-13,1,-10])],
    "interval" : 0.1,
    "bounds" : {'x': (-50, 50), 'y': (-50, 50), 'z': (-50, 50)},
    "probability" : 0.975,
    "initial_number_of_lines" : 10,
    "noise" : 0.02
}

test3 = {
    "variance" : 0.1,
    "lines" : [(([4.5,0,0], [4.5,0,3.6576])), ([-6.5, 0, 0], [-6.5,0,3.6576]),
                (([4.5,9.144,0], [4.5,9.144,3.6576])), ([-6, 9.144, 0], [-6,9.144,3.6576]),
                (([5.5,18.288,0], [5.5,18.288,3.6576])), ([-5.5, 18.288, 0], [-5.5,18.288,3.6576]),
                (([6.1,27.432,0], [6.1,27.432,3.6576])), ([-4.9, 27.432, 0], [-4.9,27.432,3.6576]),
                (([6.7,36.576,0], [6.7,36.576,3.6576])), ([-3.8, 36.576, 0], [-3.8,36.576,3.6576]),
                (([4.5,-9.144,0], [4.5,-9.144,3.6576])), ([-6.5, -9.144, 0], [-6.5,-9.144,3.6576]),
                (([4.5,-18.288,0], [4.5,-18.288,3.6576])), ([-6.5, -18.288, 0], [-6.5,-18.288,3.6576]),
                (([4.5,-27.432,0], [4.5,-27.432,3.6576])), ([-6.5, -27.432, 0], [-6.5,-27.432,3.6576]),
                (([4.5,-36.576,0], [4.5,-36.576,3.6576])), ([-6.5, -36.576, 0], [-6.5,-36.576,3.6576]),  #lightpoles
                (([3,3,0], [3,3,0.7])), (([-5,-15,0], [-5,-15,0.7])), (([4,-30,0], [4,-30,0.7])), (([7,-30,0], [7,-30,0.7])),#small pole structures such as fire hydrants
                (([-9,-5,0], [-9,-5,6])) #large pole structure such as billboards
                ],
    "interval" : 0.01,
    "bounds" : {'x': (-10, 10), 'y': (-40, 40), 'z': (-2, 10)},
    "probability" : 0.975,
    "initial_number_of_lines" : 5,
    "noise" : 0.02
}


tests_arr = [test1, test2, test3]

def generate_ptcld_from_test(test):
    lines = []
    ptclds = []
    for l in test['lines']:
        l_3d = Line3D(l[0], l[1])
        lines.append(l_3d)
        p = l_3d.generate_pts_from_line(test['interval'], test['variance'])
        ptclds.append(np.array(p))
    
    ptcld_comb = np.concatenate(tuple(ptclds))
    n_rand = int(np.ceil(test['noise']*ptcld_comb.shape[0]))
    x_rand = (test['bounds']['x'][1] - test['bounds']['x'][0]) * np.random.random_sample(n_rand) + test['bounds']['x'][0]
    y_rand = (test['bounds']['y'][1] - test['bounds']['y'][0]) * np.random.random_sample(n_rand) + test['bounds']['y'][0]
    z_rand = (test['bounds']['z'][1] - test['bounds']['z'][0]) * np.random.random_sample(n_rand) + test['bounds']['z'][0]
    rand_noise = np.stack((x_rand,y_rand,z_rand), axis=-1)
    ptcld_comb = np.concatenate((ptcld_comb, rand_noise), axis=0)

    return ptcld_comb, lines

def run_KLinesCluster(test_data, iterations = 20, output_test_data = False, verbose = False):
    ptcld, lines = generate_ptcld_from_test(test_data)
    
    if output_test_data:
        np.savetxt('ptcld.txt', np.array(ptcld), delimiter = ',')
    classifier = KLinesCluster(ptcld, test_data['variance'], test_data["probability"], test_data["initial_number_of_lines"])
    
    start = time.time()
    if verbose:
        for i in range(iterations):
            fig = plt.figure(10+i) 
            ax = fig.add_subplot(111,projection='3d') 
            ax.axes.set_xlim3d(left=test_data['bounds']['x'][0], right=test_data['bounds']['x'][1]) 
            ax.axes.set_ylim3d(bottom=test_data['bounds']['y'][0], top=test_data['bounds']['y'][1]) 
            ax.axes.set_zlim3d(bottom=test_data['bounds']['z'][0], top=test_data['bounds']['z'][1])

            for l in classifier.lines:
                l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
                if not len(classifier.line_pointclouds[l]) == 0:
                    points = Points(classifier.line_pointclouds[l])
                    points.plot_3d(ax, c='b', depthshade=False, s=0.1)
            
            if not classifier.unused_points == []:
                unused_points = Points(classifier.unused_points)
                unused_points.plot_3d(ax, c='k',depthshade=False,s=0.1)
            
            ax.set_title("iteration "+str(i))
            
            classifier.fit() 

    else:
        classifier.fit(iterations)
        
    end = time.time()
    print("Time taken:",round(end - start), " seconds")
    
    fig = plt.figure(1) 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=test_data['bounds']['x'][0], right=test_data['bounds']['x'][1]) 
    ax.axes.set_ylim3d(bottom=test_data['bounds']['y'][0], top=test_data['bounds']['y'][1]) 
    ax.axes.set_zlim3d(bottom=test_data['bounds']['z'][0], top=test_data['bounds']['z'][1])

    for l in classifier.lines:
        points = Points(classifier.line_pointclouds[l])
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
        points.plot_3d(ax, c='b', depthshade=False, s=0.1)
    
    if not classifier.unused_points == []:
        unused_points = Points(classifier.unused_points)
        unused_points.plot_3d(ax, c='k',depthshade=False,s=0.1)

    ax.set_title("final output with ptcld")
   
    fig = plt.figure(2) 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=test_data['bounds']['x'][0], right=test_data['bounds']['x'][1]) 
    ax.axes.set_ylim3d(bottom=test_data['bounds']['y'][0], top=test_data['bounds']['y'][1]) 
    ax.axes.set_zlim3d(bottom=test_data['bounds']['z'][0], top=test_data['bounds']['z'][1])

    for l in classifier.lines:
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
    ax.set_title("final output without ptcld")

def visualize_test(test):
    ptcld, lines = generate_ptcld_from_test(test)
    fig = plt.figure(0) 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=test['bounds']['x'][0], right=test['bounds']['x'][1]) 
    ax.axes.set_ylim3d(bottom=test['bounds']['y'][0], top=test['bounds']['y'][1]) 
    ax.axes.set_zlim3d(bottom=test['bounds']['z'][0], top=test['bounds']['z'][1])

    #plot ground truth lines in red:
    for l in lines:
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='k')

    ax.set_title("ground truth")
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog = 'test_lidarlinefinder',
                    description = 'runs a series of tests to test the line classifier')
    parser.add_argument('test')
    parser.add_argument('-i', '--iterations', required = False, default=20)
    parser.add_argument('-v', '--verbose', action='store_true')  
    parser.add_argument('-o', '--output', action='store_true')
    args = parser.parse_args()

    t = tests_arr[int(args.test)-1]
    i = int(args.iterations)
    o = args.output
    v = args.verbose
    
    run_KLinesCluster(t, iterations = i, output_test_data = o, verbose = v)
    visualize_test(t)

    plt.show()
    