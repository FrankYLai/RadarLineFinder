from skspatial.objects import Line, Points, LineSegment
import scipy.stats
from skspatial.plotting import plot_3d
import numpy as np
import math
import matplotlib.pyplot as plt
import time

class Line3D:
    def __init__(self, start, end):
        self.e = np.array([0, 0, 0])
        self.s = np.array([0, 0, 0])
        self.update_line(start, end)
    

    def get_length_direction(self):
        diff = np.subtract(self.e, self.s)
        length = np.linalg.norm(diff)
        diff = diff/length
        return  length, diff

    def update_line(self, start, end):
        if type(start) == list:
            self.s = np.array(start)
        else:
            self.s = start

        if type(end) == list:
            self.e = np.array(end)
        else:
            self.e = end

        self.length, self.direction = self.get_length_direction()
        self.skLine = Line(self.s, self.direction)
    
    def generate_pts_from_line(self, density, v):
        total_points = round(self.length/density)
        points = np.zeros((total_points,3))
        mean = [0, 0, 0]
        cov = np.identity(3) * v
        noise = np.random.multivariate_normal(mean,cov,total_points)

        for i in range(total_points):
            points[i] = self.direction*i*density + self.s + noise[i]
        
        return Points(points)

class KNN:
    def __init__(self, pointcloud, variance, probability = 0.9, num_lines = 1) -> None:
        
        self.total_iterations = 0
        self.pointcloud = pointcloud
        
        self.lines = []
        self.line_pointclouds = {}
        self.unused_points = []

        self.v = variance
        self.prob = probability

        self.nd = scipy.stats.norm(0, math.sqrt(variance))
        for i in range(num_lines):
            self.lines.append(self.init_line())
            self.line_pointclouds[self.lines[-1]] = np.array([])

    def init_line(self):
        selection = np.random.choice(self.pointcloud.shape[0], 2, replace = False)
        return Line3D(self.pointcloud[selection[0]], self.pointcloud[selection[1]])

    def get_dist(self, point, line):

        p = line.skLine.project_point(point)

        vec_s = p-line.s
        vec_e = p-line.e
        d_product = np.dot(vec_s, vec_e)
        if d_product< 0:
            dist = line.skLine.distance_point(point)
        else:
            dist = min(np.linalg.norm(point-line.s), np.linalg.norm(point-line.e))
        return dist
    
    def include_criteria(self, dist, line, point):
        prob = self.nd.pdf(dist)
        if prob > 1-self.prob:
            return True
        else:
            return False

    def fit(self, iterations = 1):
        for i in range(iterations):
            self.point_segmentation()
            self.update_lines()
            self.total_iterations += 1
    
    def point_segmentation(self):
        for ptcld in self.line_pointclouds:
            self.line_pointclouds[ptcld] = []
        self.unused_points = []
        for p in self.pointcloud:
            dists = []
            for i, l in enumerate(self.lines):
                dists.append(self.get_dist(p, l))
            
            tmp = min(dists)
            if self.include_criteria(tmp, self.lines[dists.index(tmp)], p):
                self.line_pointclouds[self.lines[dists.index(tmp)]].append(p)
            else:
                self.unused_points.append(p)
    
    def update_lines(self):
        for line in self.line_pointclouds:
            points = Points(self.line_pointclouds[line])
            lobf = Line.best_fit(points)
            
            dists = lobf.transform_points(points)
            min_dist = min(dists)
            max_dist = max(dists)
            start = lobf.point+lobf.direction*min_dist
            end = lobf.point+lobf.direction*max_dist

            line.update_line(start, end)
            
Variance = 1
line = Line3D([30,0,0], [0,0,0])
p = line.generate_pts_from_line(0.1, Variance)

classifier = KNN(p, Variance, 0.9, 1)
for i in range(10):
    plt.figure(i)
    fig = plt.figure() 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=-5, right=35) 
    ax.axes.set_ylim3d(bottom=-10, top=10) 
    ax.axes.set_zlim3d(bottom=-10, top=10) 
    classifier.fit()
    line = classifier.lines[0]
    points = Points(classifier.line_pointclouds[classifier.lines[0]])
    unused_points = Points(classifier.unused_points)
    line.skLine.plot_3d(ax, t_1=0, t_2=line.length, c='y')
    points.plot_3d(ax, c='r', depthshade=False)
    unused_points.plot_3d(ax, c='b',depthshade=False)
    print(line.length, line.s, line.e)


plt.show()