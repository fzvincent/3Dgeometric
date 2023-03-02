import open3d as o3d

from geomfitty import geom3d, fit3d
import geomfitty.plot
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from geomfitty import derive3d

objs = []
# objs.append(geom3d.Cylinder([0, 0, 0], [1, 0, 0], 1, 10))
# objs.append(geom3d.Cylinder([0, 5, 6], [0, 1, 0], 1,10))
spaceMax = 100
minRadius = 5
while 1:
    anchors = []
    directions = []
    anchors.append(np.random.randint(spaceMax, size=3))
    directions.append(np.random.randint(spaceMax, size=3))
    anchors.append(np.random.randint(spaceMax, size=3))
    directions.append(np.random.randint(spaceMax, size=3))
    n = np.cross(directions[0], directions[1])
    distance = np.matmul(n, anchors[1] - anchors[0]) / np.linalg.norm(n)
    if distance > 2 * minRadius * 3:
        break

Lines = [geom3d.Cylinder(i, j, 5, 200) for i, j in zip(anchors, directions)]
# Lines=[]
# Lines.append(geom3d.Cylinder([0,0,0],[1,0,0],5,100))
# Lines.append(geom3d.Cylinder([0,0,40],[0,1,0],5,100))
pTorus = list(derive3d.partialTorus_from_lines(Lines[0], Lines[1], 5))
objs = Lines + pTorus
# objs.append(geom3d.partialTorus([0, 0, 0], [0, 0, 1], 3, 1, 0, np.pi / 2))
# obj=[geom3d.partialTorus([0, 0, 0], [0, 0, 1], 3, 1, -1, np.pi / 2)]
# objs=[geom3d.partialTorus([0, 0, 0], [0, 0, 1], 3, 1, -1, np.pi / 2)]

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)
# custom_draw_geometry_with_rotation(objs)
geomfitty.plot.plot(objs, display_coordinate_frame=True)
