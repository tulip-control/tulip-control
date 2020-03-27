import math
from shapely.geometry import Polygon
from descartes import PolygonPatch


class VehicleState:
    """A class for defining SE2 state of a rectangular vehicle
    """

    front_length = 3.0
    rear_length = 1.5
    half_width = 0.8

    def __init__(self, configuration):
        self.x = configuration[0]
        self.y = configuration[1]
        self.theta = configuration[2]
        self.configuration = configuration
        self._compute_footprint()

    def _compute_footprint(self):
        corners = [
            (
                self.x
                + self.half_width * math.sin(self.theta)
                + self.front_length * math.cos(self.theta),
                self.y
                - self.half_width * math.cos(self.theta)
                + self.front_length * math.sin(self.theta),
            ),
            (
                self.x
                - self.half_width * math.sin(self.theta)
                + self.front_length * math.cos(self.theta),
                self.y
                + self.half_width * math.cos(self.theta)
                + self.front_length * math.sin(self.theta),
            ),
            (
                self.x
                - self.half_width * math.sin(self.theta)
                - self.rear_length * math.cos(self.theta),
                self.y
                + self.half_width * math.cos(self.theta)
                - self.rear_length * math.sin(self.theta),
            ),
            (
                self.x
                + self.half_width * math.sin(self.theta)
                - self.rear_length * math.cos(self.theta),
                self.y
                - self.half_width * math.cos(self.theta)
                - self.rear_length * math.sin(self.theta),
            ),
        ]
        self.footprint = Polygon(corners)

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y, self.theta - other.theta)

    def __str__(self):
        return "({},{},{})".format(self.x, self.y, self.theta)

    def plot(self, ax, zorder, draw_heading=False):
        patch = PolygonPatch(self.footprint, alpha=0.5, zorder=2)
        ax.add_patch(patch)
        if draw_heading:
            ax.plot([self.x], [self.y], "r*", zorder=2)
            ax.plot(
                [self.x, self.x + self.front_length * math.cos(self.theta)],
                [self.y, self.y + self.front_length * math.sin(self.theta)],
                "r-",
                zorder=2,
            )
        xbound = (self.xmin(), self.xmax())
        ybound = (self.ymin(), self.ymax())
        return (xbound, ybound)

    def dist(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def xmin(self):
        return self.footprint.bounds[0]

    def xmax(self):
        return self.footprint.bounds[2]

    def ymin(self):
        return self.footprint.bounds[1]

    def ymax(self):
        return self.footprint.bounds[3]

    def __gt__(self, obj):
        return (self.x, self.y, self.theta) > (obj.x, obj.y, obj.theta)

    def __le__(self, obj):
        return not (self.__gt__(obj))
