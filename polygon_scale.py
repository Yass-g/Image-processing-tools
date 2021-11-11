from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

import itertools as IT
def scale_polygon(path,offset):
    center = centroid_of_polygon(path)
    for i in path:
        if i[0] > center[0]:
            i[0] = offset*(i[0]-center[0])+center[0]
        else:
            i[0] = offset*(i[0]-center[0])+center[0]
        if i[1] > center[1]: 
            i[1] = offset*(i[1]-center[1])+center[1]
        else:
            i[1] = offset*(i[1]-center[1])+center[1]
    #i[0] = np.sqrt(offset)*(i[0]-center[0])+center[0]
    #i[1] = np.sqrt(offset)*(i[1]-center[1])+center[1]
    print(path)
    return path


def area_of_polygon(x, y):
    area = 0.0
    for i in range(-1, len(x) - 1):
        print(x[i],y[i])
        area += x[i] * (y[i + 1] - y[i - 1])
    return area / 2.0

def centroid_of_polygon(points):
    x = []
    y = []
    for e in points:
        x+=[e[0]]
        y+=[e[1]]
    print(x)
    print(y)
    area = area_of_polygon(x,y)
    print(area)
    result_x = 0
    result_y = 0
    N = len(points)
    points = IT.cycle(points)
    x1, y1 = next(points)
    for i in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return (result_x, result_y)


h = 200
w = 100
plt.axis('equal')
#frame = np.array([[0.0,0.0],[0.0,200.0],[100.0,200.0],[100.0,0.0]])
points = np.array([[50.0,60.0],[70.0,60.0],[70.0,80.0],[50.0,80.0]])
print(points)
hull = ConvexHull(points)
#plt.fill(frame[:,0], frame[:,1], color = (0.0, 0.0, 0.0))
#plt.plot(points[:,0], points[:,1], 'o')
#for simplex in hull.simplices:
#    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.fill(points[:,0], points[:,1], color = (0.6, 0.6, 0.6))
print(centroid_of_polygon(points))
scaled = scale_polygon(points,0.9)
#for simplex in hull.simplices:
#    plt.plot(scaled[simplex, 0], scaled[simplex, 1], 'k-')

plt.fill(scaled[:,0], scaled[:,1], "white")
figure = plt.gcf()
figure.set_size_inches(8, 6)
axes = plt.gca()

plt.ylim([0,200])
plt.xlim([0,100])
plt.axis('equal')
#plt.axis('off')
#plt.savefig("sample.png", facecolor='black')
plt .show()
