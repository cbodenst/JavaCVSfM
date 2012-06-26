from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

points = open('points.txt','r').read();
points=points.splitlines()
#print points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x=[]
y=[]
z=[]
for ps in points:
    p = ps.split("\t")
    x.append(float(p[0]))
    y.append(float(p[1]))
    z.append(float(p[2]))

n = 100
ax.scatter(x,z,y, marker='o')
plt.ylim([0,100])
plt.xlim([0,100])
#plt.zlim([0,100])
#x = [6,3,6,9,12,24]
#y = [3,5,78,12,23,56]
#z = [3,5,78,12,23,56]



#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#    xs = randrange(n, 23, 32)
#    ys = randrange(n, 0, 100)
#    zs = randrange(n, zl, zh)
#    ax.scatter(xs, ys, zs, c=c, marker=m)
plt.show()
