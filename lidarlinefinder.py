from skspatial.objects import Line, Points, LineSegment
import scipy.stats
from skspatial.plotting import plot_3d
import numpy as np
import math
import matplotlib.pyplot as plt
import time
'''
TODO:
    ideas: 
        1. prune lines that are too close to eachother
        2. after 10 iterations, add everything to the same cluster and try to do clustering again
        3. seperate line into segments if the point projections onto like line using transform_points()
'''


# np.random.seed(69)


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
        # selection = np.random.choice(self.pointcloud.shape[0], 2, replace = False)
        # return Line3D(self.pointcloud[selection[0]], self.pointcloud[selection[1]])
        return Line3D(self.pointcloud[-1], self.pointcloud[30])

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
            self.segment_line()
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
    
    def segment_line(self):
        extra_lines = []
        new_lines = {}
        for line in self.line_pointclouds:
            np_lindist = line.skLine.transform_points(self.line_pointclouds[line])
           
            lin_dists = list(np_lindist) #length of N
            sorted_lindist = np.sort(np_lindist)
            
            np_offset_lindist = sorted_lindist[1:]
            sorted_sublist = sorted_lindist[0:len(sorted_lindist)-1]
        
            diffs = np.subtract(np_offset_lindist,sorted_sublist) # length of N-1
            avg_diff = np.average(np.array(diffs))

            # print(lin_dists)
            # print(diffs)
            # print(avg_diff)

            line_segments = []
            line_segments.append(0)
            for idx, diff in enumerate(diffs):
                if diff > 2 * avg_diff and diff > 2*self.v: # if the next point is twice the average away from the previous point, segment
                    line_segments.append(idx+1)
                    print("segmentation at index", idx+1, 'of', len(lin_dists))
            line_segments.append(len(lin_dists))

            if len(line_segments) > 2:
                new_line_generated = False
                for split in range(len(line_segments)-1):
                    ptcld = []
                    for p in range(line_segments[split], line_segments[split+1]):
                        #append a single point
                        ptcld.append(self.line_pointclouds[line][lin_dists.index(sorted_lindist[p])]) #finds the point using the index given by find
                    
                    #requires at least 3 points to define a line of best fit
                    if len(ptcld)<3:
                        continue
                    new_line_generated = True
                    #generate new LOBF based on the new pointcloud:
                    points = Points(ptcld)
                    lobf = Line.best_fit(points)
                    
                    d = lobf.transform_points(points)
                    min_dist = min(d)
                    max_dist = max(d)
                    start = lobf.point+lobf.direction*min_dist
                    end = lobf.point+lobf.direction*max_dist

                    l = Line3D(start, end)
                    self.lines.append(l)
                    new_lines[l] = ptcld # save this so we can add it to the class dictionary after

                if new_line_generated:
                    extra_lines.append(line)
            
        #delete old lines
        for l in extra_lines:
            del self.line_pointclouds[l]
            self.lines.remove(l)
        for l in new_lines:
            self.line_pointclouds[l] = new_lines[l]

            
Variance = 1
line = Line3D([30,0,0], [0,0,0])
line2 = Line3D([5, 5, 3], [5,9,8])
p = line.generate_pts_from_line(0.333, Variance)
p2 = line2.generate_pts_from_line(0.3333, Variance)



pointcloud = np.concatenate((np.array(p), np.array(p2)))
# pointcloud = np.array(p)
pointcloud = Points(pointcloud)
fig = plt.figure(0)
ax = fig.add_subplot(111,projection='3d') 
pointcloud.plot_3d(ax, c='b',depthshade=False)

classifier = KNN(pointcloud, Variance, 0.975, 1)
fig = plt.figure(1) 
ax = fig.add_subplot(111,projection='3d') 
ax.axes.set_xlim3d(left=-5, right=35) 
ax.axes.set_ylim3d(bottom=-10, top=10) 
ax.axes.set_zlim3d(bottom=-10, top=10)
for l in classifier.lines:
    l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
pointcloud.plot_3d(ax, c='b',depthshade=False)
for i in range(1,20):
    plt.figure(i)
    fig = plt.figure() 
    ax = fig.add_subplot(111,projection='3d') 
    ax.axes.set_xlim3d(left=-5, right=35) 
    ax.axes.set_ylim3d(bottom=-10, top=10) 
    ax.axes.set_zlim3d(bottom=-10, top=10) 
    classifier.fit()
    for l in classifier.lines:
        points = Points(classifier.line_pointclouds[l])
        l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
        points.plot_3d(ax, c='r', depthshade=False)
        print(l.length, l.s, l.e)
    unused_points = Points(classifier.unused_points)
    unused_points.plot_3d(ax, c='b',depthshade=False)



plt.show()