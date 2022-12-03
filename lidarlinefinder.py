from skspatial.objects import Line, Points, LineSegment
import scipy.stats
from skspatial.plotting import plot_3d
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import itertools


class Line3D:
    def __init__(self, start, end):
        self.e = np.array([0, 0, 0])
        self.s = np.array([0, 0, 0])
        self.update_line(start, end)
    
    
    def get_length_direction(self):
        """
        get_length_direction: 
        returns the length and direction of the current line
        self.e and self.s must be created

        :return: length and direction of the line
        """ 
        diff = np.subtract(self.e, self.s)
        length = np.linalg.norm(diff)
        diff = diff/length
        return  length, diff

    def update_line(self, start, end):
        """
        update_line updates the location of a line given start and end points

        :start: starting point of the line
        :end: ending point of the line

        :return: None
        """ 
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
    
    def generate_pts_from_line(self, interval, v):
        """
        generate_pts_from_line generates points describing the line object

        :interval: determines the interval that the points get generated in
        :v: ending point of the line

        :return: None
        """ 
        total_points = round(self.length/interval)
        points = np.zeros((total_points,3))
        mean = [0, 0, 0]
        cov = np.identity(3) * v
        noise = np.random.multivariate_normal(mean,cov,total_points)

        for i in range(total_points):
            points[i] = self.direction*i*interval + self.s + noise[i]
        
        return Points(points)

    def get_dist(self, point):
        """
        get_dist gets the distance from a point to the line segment

        :point: the point to be compared to

        :return: distance of the point to the line segment
        """ 

        p = self.skLine.project_point(point)

        vec_s = p-self.s
        vec_e = p-self.e
        d_product = np.dot(vec_s, vec_e)
        if d_product < 0:
            dist = self.skLine.distance_point(point)
        else:
            dist = min(np.linalg.norm(point-self.s), np.linalg.norm(point-self.e))
        return dist

    def isclose(self, other):
        """
        isclose Determines if another line is close to the main line

        :other: another line

        :return: distance of the point to the line segment
        """ 
        return (np.allclose(self.direction, other.direction,atol=1) and (self.get_dist(other.s) < 1 or self.get_dist(other.e) < 1)) or (self.get_dist(other.s) < 1 and self.get_dist(other.e) < 1)
    
    def __eq__(self,other):
        return isinstance(other,Line3D) and np.all(self.s == other.s) and np.all(self.e == other.e)

    def __hash__(self):
        return id(self)

class KNN:
    def __init__(self, pointcloud, variance, probability = 0.9, num_lines = 1) -> None:
        
        self.total_iterations = 0
        self.pointcloud = pointcloud
        
        self.lines = []
        self.line_pointclouds = {}
        self.unused_points = []

        self.v = variance
        self.prob = probability
        
        self.used_points = []

        self.nd = scipy.stats.norm(0, math.sqrt(variance))
        for i in range(num_lines):
            self.lines.append(self.init_line())
            self.line_pointclouds[self.lines[-1]] = np.array([])
        
        self.used_points = self.tally_used_points()

    def init_line(self):
        """
        init_line Initializes a line segment in the space. the line segment is created by 
                  sampling two random points and using them as start and end points.  

        :return: a Line3D object representing the line just generated
        """ 
        selection = np.random.choice(self.pointcloud.shape[0], 2, replace = False)
        return Line3D(self.pointcloud[selection[0]], self.pointcloud[selection[1]])
        # return Line3D(self.pointcloud[-1], self.pointcloud[30])

    def get_dist(self, point, line):
        """
        get_dist gets the distance from a point to the line segment

        :point: the point to be compared to
        :line: the line being compared to

        :return: distance of the point to the line segment
        """ 
        p = line.skLine.project_point(point)

        vec_s = p-line.s
        vec_e = p-line.e
        d_product = np.dot(vec_s, vec_e)
        if d_product < 0:
            dist = line.skLine.distance_point(point)
        else:
            dist = min(np.linalg.norm(point-line.s), np.linalg.norm(point-line.e))/2
        return dist
    
    def include_criteria(self, dist):
        """
        include_criteria calculates the probability of a point with dist to be generated from the line
                         using variance. the probability is defined on initialization and is based.

        :dist: distacne fromt he line to the point

        :return: if the point is within probability tolerance
        """ 
        prob = self.nd.pdf(dist)
        if prob > 1-self.prob:
            return True
        else:
            return False
    
    def tally_used_points(self):
        """
        tally_used_points sums up the number of points in the pointcloud for each line.

        :return: array for each line being tracked by the classifier representing number of used points in the tally.
        """ 
        used_points = []
        for l in self.line_pointclouds:
            used_points.append(len(self.line_pointclouds[l]))
        return used_points
    
    def check_lines_settle(self):
        """
        check_lines_settle checks to see if the lines have stabilized

        :return: True if the lines are stabilized and have not updated
        """ 
        used_points = self.tally_used_points()
        if len(used_points) != len(self.used_points):
            return False
        
        sum = 0
        for i in range(len(used_points)):
            diff = abs(self.used_points[i]- used_points[i])
            if diff>2:
                return False
        #lines have settled if number of lines generated is the same, 
        # and the difference bettwen this iteration of used points and the next iteration of used poits
        # is less than 2 for each line
        return True
            
    def start_end_from_ptcld(self, ptcld):
        """
        start_end_from_ptcld generates start and ending points of the line of best fit
                            from pointcloud cluster

        :ptcld: the pointcloud to be processed.

        :return: start and endpoint of a line of best fit generated with this pointcloud
        """ 
        points = Points(ptcld)
        lobf = Line.best_fit(points)
        
        dists = lobf.transform_points(points)
        min_dist = min(dists)
        max_dist = max(dists)
        start = lobf.point+lobf.direction*min_dist
        end = lobf.point+lobf.direction*max_dist

        return start, end

    def line_from_all_unused(self):
        """
        line_from_all_unused generates a line of best fit from all the unused points in the 
                             pointcloud. This new line is added to self.lines

        :return: None
        """
        if len(self.unused_points) == 0:
            return
        start, end = self.start_end_from_ptcld(self.unused_points)
        l = Line3D(start, end)
        self.lines.append(l)
        self.line_pointclouds[l] = []


    def fit(self, iterations = 1):
        """
        fit iterates the classifier to find better lines of best fit

        :iterations: the number of iterations to iterate through

        :return: None
        """
        for i in range(iterations):
            self.total_iterations += 1

            self.point_segmentation()
            self.update_lines()
            self.segment_line()
            self.prune_lines()
            if self.check_lines_settle() and len(self.unused_points)/sum(self.used_points)>0.2:
                print("_______________generate new line_______________")
                self.line_from_all_unused()
                self.point_segmentation()
                self.update_lines()
                self.segment_line()
            self.used_points = self.tally_used_points()
            print(self.used_points)
    
    def point_segmentation(self):
        """
        point_segmentation checks the pointcloud against each line and associates points that are meet
                           proximity criteria into the line_pointcloud

        :return: None
        """
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
        """
        update_lines generates lines from line_pointclouds using the line of best fit method

        :return: None
        """
        destructable_lines = []
        for line in self.line_pointclouds:
            # points = Points(self.line_pointclouds[line])
            # lobf = Line.best_fit(points)
            
            # dists = lobf.transform_points(points)
            # min_dist = min(dists)
            # max_dist = max(dists)
            # start = lobf.point+lobf.direction*min_dist
            # end = lobf.point+lobf.direction*max_dist
            if len(self.line_pointclouds[line]) < 10:
                destructable_lines.append(line)
            else:
                start,  end = self.start_end_from_ptcld(self.line_pointclouds[line])

                line.update_line(start, end)
        for l in destructable_lines:
            self.destruct_line(l)
    
    def destruct_line(self, l):
        """
        destruct_line deletes a line

        :return: None
        """
        self.lines.remove(l)
        del self.line_pointclouds[l]

    
    def segment_line(self):
        """
        segment_line segments a line into multiple smaller lines if the points defining a line are of seperate
                     clusters. The criteria for segmenting is the distance between adjacent points must be much much greater
                     than the average distance of the points and also greater than a threshold scaled by variance.

        :return: None
        """
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
                if diff > 3 * avg_diff and diff > 3*self.v: # if the next point is twice the average away from the previous point, segment
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
                    # points = Points(ptcld)
                    # lobf = Line.best_fit(points)
                    
                    # d = lobf.transform_points(points)
                    # min_dist = min(d)
                    # max_dist = max(d)
                    # start = lobf.point+lobf.direction*min_dist
                    # end = lobf.point+lobf.direction*max_dist
                    start, end = self.start_end_from_ptcld(ptcld)

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

    def prune_lines(self):
        """
        prune_lines deletes lines that are to close to eachother @nick fill in details

        :return: None
        """
        del_list = []
        chosen_line = None
        for (ix, x), (iy, y) in itertools.combinations(enumerate(self.line_pointclouds),2):
            if x == y:
                print("2 lines are equal")
                if chosen_line == None:
                    chosen_line = x
                    if y not in del_list:
                        del_list.append(y)
                elif chosen_line == x:
                    if y not in del_list:
                        del_list.append(y)

            elif x.isclose(y):
                print("Lines are close")
                if chosen_line == None:
                    chosen_line = x
                    if y not in del_list:
                        del_list.append(y)
                elif chosen_line == x:
                    if y not in del_list:
                        del_list.append(y)
        for item in del_list:
            self.line_pointclouds[chosen_line].extend(self.line_pointclouds[item])
            self.destruct_line(item)
            start, end = self.start_end_from_ptcld(self.line_pointclouds[chosen_line])
            chosen_line.update_line(start, end)

            
Variance = 0.5
line = Line3D([30,0,0], [0,0,0])
line2 = Line3D([5, 5, 3], [5,9,8])
line3 = Line3D([20, -9, -3], [25,-3,-9])

line4 = Line3D([15,10,0], [20,35,0])
line5 = Line3D([-10, 3, 40], [-2,3,25])
line6 = Line3D([2, -20, -13], [-13,1,-10])
p = line.generate_pts_from_line(0.1, Variance)
p2 = line2.generate_pts_from_line(0.1, Variance)
p3 = line3.generate_pts_from_line(0.1, Variance)

p4 = line4.generate_pts_from_line(0.1, Variance)
p5 = line5.generate_pts_from_line(0.1, Variance)
p6 = line6.generate_pts_from_line(0.1, Variance)


pointcloud = np.concatenate((np.array(p), np.array(p2),np.array(p3),np.array(p4),np.array(p5), np.array(p6)))
# pointcloud = np.array(p)
pointcloud = Points(pointcloud)
fig = plt.figure(0)
ax = fig.add_subplot(111,projection='3d') 
pointcloud.plot_3d(ax, c='b',depthshade=False)

classifier = KNN(pointcloud, Variance, 0.975, 1)
# for i in range(1,20):
    # plt.figure(i)
    # fig = plt.figure() 
    # ax = fig.add_subplot(111,projection='3d') 
    # ax.axes.set_xlim3d(left=-50, right=50) 
    # ax.axes.set_ylim3d(bottom=-50, top=50) 
    # ax.axes.set_zlim3d(bottom=-50, top=50) 
classifier.fit(20)
    # for l in classifier.lines:
    #     points = Points(classifier.line_pointclouds[l])
    #     l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
    #     #points.plot_3d(ax, c='r', depthshade=False)
    #     print(l.length, l.s, l.e)
    # if not classifier.unused_points == []:
    #     unused_points = Points(classifier.unused_points)
    #     unused_points.plot_3d(ax, c='b',depthshade=False)


fig = plt.figure(100) 
ax = fig.add_subplot(111,projection='3d') 

ax.axes.set_xlim3d(left=-50, right=50) 
ax.axes.set_ylim3d(bottom=-50, top=50) 
ax.axes.set_zlim3d(bottom=-50, top=50) 
for l in classifier.lines:
    points = Points(classifier.line_pointclouds[l])
    l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')

fig = plt.figure(110) 
ax = fig.add_subplot(111,projection='3d') 

ax.axes.set_xlim3d(left=-50, right=50) 
ax.axes.set_ylim3d(bottom=-50, top=50) 
ax.axes.set_zlim3d(bottom=-50, top=50) 

for l in classifier.lines:
    points = Points(classifier.line_pointclouds[l])
    l.skLine.plot_3d(ax, t_1=0, t_2=l.length, c='y')
    points.plot_3d(ax, c='r', depthshade=False)
    print(l.length, l.s, l.e)

if not classifier.unused_points == []:
    unused_points = Points(classifier.unused_points)
    unused_points.plot_3d(ax, c='b',depthshade=False)

fig = plt.figure(120) 
ax = fig.add_subplot(111,projection='3d') 

ax.axes.set_xlim3d(left=-50, right=50) 
ax.axes.set_ylim3d(bottom=-50, top=50) 
ax.axes.set_zlim3d(bottom=-50, top=50)
line.skLine.plot_3d(ax, t_1=0, t_2=line.length, c='y')
line2.skLine.plot_3d(ax, t_1=0, t_2=line2.length, c='y')
line3.skLine.plot_3d(ax, t_1=0, t_2=line3.length, c='y')
line4.skLine.plot_3d(ax, t_1=0, t_2=line4.length, c='y')
line5.skLine.plot_3d(ax, t_1=0, t_2=line5.length, c='y')
line6.skLine.plot_3d(ax, t_1=0, t_2=line6.length, c='y')


plt.show()
