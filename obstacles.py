from shapely.geometry import LineString, Polygon, Point
import numpy as np

def check_collision(segments, obstacles):
    """Controlla se i segmenti del braccio collidono con almeno un ostacolo"""
    for (p0, p1) in segments:
        arm_segment = LineString([p0, p1])
        for obs in obstacles:
            if arm_segment.intersects(obs):
                return True
    return False

def make_rect(xmin, xmax, ymin, ymax):
    """Costruisce un rettangolo come poligono shapely"""
    return Polygon([(xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)])

def make_circle(xc, yc, r, n_points=20):
    """Costruisce un cerchio come poligono shapely"""
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    points = [(xc + r*np.cos(a), yc + r*np.sin(a)) for a in angles]
    return Polygon(points)

def make_polygon(vertices):
    """Costruisce un poligono arbitrario"""
    return Polygon(vertices)