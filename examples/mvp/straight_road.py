# Straight road setup for overtaking.py example

from shapely.geometry import Polygon
from descartes import PolygonPatch


class Obstacle:
    """A class for defining an obstacle and its clearance zone

    An obstacle is a rectangle, with the lower-left corner at (x0, y0)
    and upper right corner at (x0 + length, y0 + width).
    """

    lateral_clearance = 1.0
    longitudinal_clearance = 1.5

    def __init__(self, x0, y0, width, length):
        self.clearance_zone = Polygon(
            [
                (x0 + length + self.longitudinal_clearance, y0),
                (x0 + length + self.longitudinal_clearance, y0 + width),
                (x0 + length, y0 + width + self.lateral_clearance),
                (x0, y0 + width + self.lateral_clearance),
                (x0 - self.longitudinal_clearance, y0 + width),
                (x0 - self.longitudinal_clearance, y0),
                (x0, y0 - self.lateral_clearance),
                (x0 + length, y0 - self.lateral_clearance),
            ]
        )
        self.footprint = Polygon(
            [(x0 + length, y0), (x0 + length, y0 + width), (x0, y0 + width), (x0, y0)]
        )

    def plot(self, ax, zorder=2):
        clearance_zone_patch = PolygonPatch(
            self.clearance_zone, fc="red", alpha=0.5, zorder=zorder
        )
        object_patch = PolygonPatch(
            self.footprint, fc="red", ec="black", alpha=1.0, zorder=zorder + 1
        )
        ax.add_patch(clearance_zone_patch)
        ax.add_patch(object_patch)
        xbound = (self.clearance_zone.bounds[0], self.clearance_zone.bounds[2])
        ybound = (self.clearance_zone.bounds[1], self.clearance_zone.bounds[3])
        return (xbound, ybound)

    def is_in_clearance_zone(self, polygon):
        """Check whether the given polygon intersects with the clearance zone
        """
        return self.clearance_zone.intersects(polygon)

    def is_in_collision(self, polygon):
        """Check whether the given polygon intersects with this obstacle
        """
        return self.footprint.intersects(polygon)


class Road:
    """A class for defining a straight road along the x-axis with 2 lanes.

    The road is a rectangle, with the lower-left corner at (0,0) and upper right corner
    at (road_length, 2*lane_width).
    """

    def __init__(self, lane_width, road_length):
        self.lane_width = lane_width
        self.width = lane_width * 2
        self.length = road_length
        self.polygon = Polygon(
            [(0, 0), (self.length, 0), (self.length, self.width), (0, self.width)]
        )

    def plot(self, ax, zorder=0):
        road_line_width = 8
        lane_line_width = 8
        lane_line_separation = 0.3

        patch = PolygonPatch(
            self.polygon.buffer(0.0, cap_style=2, join_style=2),
            fc=(0.7, 0.7, 0.7),
            zorder=zorder,
        )
        ax.add_patch(patch)
        ax.plot(
            [0, self.length], [0, 0], "w-", linewidth=road_line_width, zorder=zorder + 1
        )
        ax.plot(
            [0, self.length],
            [self.width, self.width],
            "w-",
            linewidth=road_line_width,
            zorder=zorder + 1,
        )
        ax.plot(
            [0, self.length],
            [self.lane_width-lane_line_separation, self.lane_width-lane_line_separation],
            "w-",
            linewidth=lane_line_width,
            zorder=zorder + 1,
        )
        ax.plot(
            [0, self.length],
            [self.lane_width+lane_line_separation, self.lane_width+lane_line_separation],
            "w-",
            linewidth=lane_line_width,
            zorder=zorder + 1,
        )
        xbound = (0, self.length)
        ybound = (0, self.width)
        return (xbound, ybound)

    def contain(self, polygon):
        """Check whether the given polygon is within the road
        """
        bounds = polygon.bounds
        return (
            bounds[0] >= 0
            and bounds[1] >= 0
            and bounds[2] <= self.length
            and bounds[3] <= self.width
        )

    def is_in_lane(self, polygon):
        """Check whether the given polygon is within the bottom lane
        """
        bounds = polygon.bounds
        return (
            bounds[0] >= 0
            and bounds[1] >= 0
            and bounds[2] <= self.length
            and bounds[3] <= self.lane_width
        )
