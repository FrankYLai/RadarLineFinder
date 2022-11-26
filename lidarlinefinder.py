from skspatial.objects import Line, Points
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
  def __init__(self, pointcloud, num_lines = 1) -> None:
    self.pointcloud = pointcloud

    self.lines = []
    for i in range(num_lines):
      self.lines.append(self.init_line())
  
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

line = Line3D([30,0,0], [0,0,0])
p = line.generate_pts_from_line(0.1, 0.001)
plot_3d(
    line.skLine.plotter(t_1=0, t_2=7, c='y'),
    p.plotter(c='b', depthshade=False),

)

print(p.shape)

classifier = KNN(p, 1)
print(p[10], classifier.get_dist([-1,0,0], line))
print(line.skLine.distance_point(p[10]))

#test code to see points
points = Points(
    [
        [0, 0, 0],
        [1, 1, 0],
        [2, 3, 2],
        [3, 2, 3],
        [4, 5, 4],
        [6, 5, 5],
        [6, 6, 5],
        [7, 6, 7],
    ],
)


t1 = time.time()
line_fit = Line.best_fit(points)
print("best fit line took:", time.time()-t1)

print(line_fit.direction)
print(line_fit.point)


plot_3d(
    line_fit.plotter(t_1=0, t_2=7, c='y'),
    points.plotter(c='b', depthshade=False),

)
plt.show()