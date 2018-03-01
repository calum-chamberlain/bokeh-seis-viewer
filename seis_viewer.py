"""
Seismicity viewer written by Calum Chamberlain.

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

from obspy import read_events, Catalog
from matplotlib.path import Path

from bokeh.io import output_file, show
from bokeh.layouts import layout, widgetbox
from bokeh.plotting import figure
from bokeh.models import (
    LinearColorMapper, ColorBar, ColumnDataSource, HoverTool)
from bokeh.models.widgets import Slider
from bokeh.palettes import Viridis256
from bokeh.tile_providers import STAMEN_TERRAIN


class Origin(object):
    """
    Holder for origin information.
    """
    def __init__(self, latitude, longitude, depth, id, time, magnitude,
                 x=None, y=None):
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.id = id
        self.time = time
        self.magnitude = magnitude
        self.x = x
        self.y = y
        if None in (self.x, self.y):
            self.x, self.y = _wgs84_to_web_mercator(
                lat=self.latitude, lon=self.longitude)

    def __repr__(self):
        print_str = (
            "ID: {0}, Latitude: {1}, Longitude: {2}, Depth: "
            "{3}m, Time: {4}, X: {5}m, Y: {6}m".format(
                self.id, self.latitude, self.longitude, self.depth, self.time,
                self.x, self.y))
        return print_str


def _wgs84_to_web_mercator(lon, lat):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    x = lon * (k * np.pi/180.0)
    y = np.log(np.tan((90 + lat) * np.pi/360.0)) * k
    return x, y


def _get_xyz_data(filename):
    """
    Parse data from obspy catalog to xyz.

    :param filename: Filename to read from, or Catalog object
    :return: list of Origin
    """
    if isinstance(filename, Catalog):
        catalog = filename
    else:
        catalog = read_events(filename)
    origins = []
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        try:
            magnitude = (
                    event.preferred_magnitude().mag or event.magnitudes[0].mag)
        except IndexError:
            magnitude = 1
        origins.append(Origin(
            latitude=origin.latitude, longitude=origin.longitude,
            depth=origin.depth / 1000, id=event.resource_id.id, time=origin.time,
            magnitude=magnitude))
    return origins


def _get_map_bounds(origins, pad=10000, z_pad=1):
    """
    Get the required map boundaries.

    :param origins: List of Origins
    :type origins: list
    :param pad: Pad to x and y in meters
    :type pad: float
    :param z_pad: Pad for vertical in km
    :type z_pad: float

    :return: dict of tuples
    """
    x_vals = [origin.x for origin in origins]
    y_vals = [origin.y for origin in origins]
    z_vals = [origin.depth for origin in origins]
    bounds = {"x_range": (min(x_vals) - pad, max(x_vals) + pad),
              "y_range": (min(y_vals) - pad, max(y_vals) + pad),
              "z_range": (min(z_vals) - z_pad, max(z_vals) + z_pad)}
    return bounds


def _get_section_bounds(map_bounds):
    """
    Get the bounds of the orthogonal cross-sections
    :param map_bounds: Dict of x and y ranges
    :return: dict of section bounds
    """
    x_min = map_bounds['x_range'][0]
    x_max = map_bounds['x_range'][1]
    x_mid = x_min + (x_max - x_min) / 2
    y_min = map_bounds['y_range'][0]
    y_max = map_bounds['y_range'][1]
    y_mid = y_min + (y_max - y_min) / 2
    return {'cross_vert': [(y_min, x_mid), (y_max, x_mid)],
            'cross_horiz': [(x_min, y_mid), (x_max, y_mid)]}


def _plot_origins_map(origins):
    """
    Plot origins scaled by magnitude and coloured by depth.

    :param origins: List of Origins
    """
    bounds = _get_map_bounds(origins=origins)
    x_section_bounds = _get_section_bounds(bounds)

    z_vals = [ori.depth for ori in origins]
    source = ColumnDataSource(dict(
        x=[ori.x for ori in origins], y=[ori.y for ori in origins], z=z_vals,
        sizes=[ori.magnitude ** 2 for ori in origins],
        magnitudes=[ori.magnitude for ori in origins],
        ids=[ori.id for ori in origins],
        origin_times=[ori.time.strftime("%Y/%m/%d %H:%M:%S")
                      for ori in origins]))

    hover = HoverTool(tooltips=[("Origin time", "@origin_times"),
                                ("Magnitude", "@magnitudes"),
                                ("Depth (km)", "@z"), ("ID", "@ids")])

    geomap = figure(tools=[hover, 'save', 'pan', 'wheel_zoom'],
                    x_range=bounds['x_range'],
                    y_range=bounds['y_range'], width=500, plot_height=500)
    geomap.axis.visible = False
    geomap.add_tile(STAMEN_TERRAIN)

    geomap.line([x_section_bounds['cross_vert'][0][1],
                 x_section_bounds['cross_vert'][1][1]],
                [x_section_bounds['cross_vert'][0][0],
                 x_section_bounds['cross_vert'][1][0]],
                line_width=5, color="black")
    geomap.line([x_section_bounds['cross_horiz'][0][0],
                 x_section_bounds['cross_horiz'][1][0]],
                [x_section_bounds['cross_horiz'][0][1],
                 x_section_bounds['cross_horiz'][1][1]],
                line_width=5, color="black")

    color_mapper = LinearColorMapper(
        palette=Viridis256, low=min(z_vals), high=max(z_vals))
    geomap.circle(x='x', y='y', color={'field': 'z', 'transform': color_mapper},
                  size={'field': 'sizes'}, name="Origins", source=source)

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12,
                         location=(0, 0), title='Depth (km)')
    geomap.add_layout(color_bar, 'left')
    return geomap


def _extract_section(origins, cross_section, swath_width):
    """
    Get the locations that fall close to a cross-section

    :param origins: List of Origins
    :param cross_section: list of tuples of start and stop locations
    :param swath_width: width in meters for swath
    :return: ColumnDataSource of x, z, size, where x is along the section.
    """
    try:
        angle = np.arctan((cross_section[0][0] - cross_section[1][0]) /
                          (cross_section[0][1] - cross_section[1][1]))
    except ZeroDivisionError:
        angle = 0
    add_x = np.sin(angle) * swath_width
    add_y = np.cos(angle) * swath_width
    corners = Path(vertices=[
        (cross_section[0][0] - add_x, cross_section[0][1] - add_y),
        (cross_section[0][0] + add_x, cross_section[0][1] + add_y),
        (cross_section[1][0] + add_x, cross_section[1][1] + add_y),
        (cross_section[1][0] - add_x, cross_section[1][1] - add_y),
        (cross_section[0][0] - add_x, cross_section[0][1] - add_y)],
        codes=[Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
               Path.CLOSEPOLY])
    x, z, size = ([], [], [])
    for origin in origins:
        if corners.contains_point((origin.y, origin.x)):
            x.append(np.sqrt((origin.x - cross_section[0][0]) ** 2 +
                             (origin.y - cross_section[0][1]) ** 2))
            z.append(origin.depth)
            size.append(origin.magnitude ** 2)
    return ColumnDataSource(dict(x=x, z=z, size=size))


def seis_viewer(filename, showfig=True):
    swath_width = 20000  # width in meters to extract
    origins = _get_xyz_data(filename)
    # Construct map
    bounds = _get_map_bounds(origins=origins)
    x_section_bounds = _get_section_bounds(bounds)
    geomap = _plot_origins_map(origins=origins)

    # Construct first cross-section
    cross_vert = figure(y_range=bounds['y_range'], x_range=bounds['z_range'],
                        width=250, plot_height=500)
    vert_source = _extract_section(
        origins=origins, cross_section=x_section_bounds['cross_vert'],
        swath_width=swath_width)
    cross_vert.circle(x='z', y='x', size={'field': 'size'}, source=vert_source)

    # Construct second cross-section
    cross_horiz = figure(y_range=bounds['z_range'], x_range=bounds['x_range'],
                         width=500, plot_height=250)
    horiz_source = _extract_section(
        origins=origins, cross_section=x_section_bounds['cross_horiz'],
        swath_width=swath_width)
    cross_horiz.circle(x='z', y='x', size={'field': 'size'},
                       source=horiz_source)

    # Define the widgets!
    cross_vert_slider = Slider(start=0, end=10, value=5, step=0.1,
                               title="Vertical cross-section position")
    cross_horiz_slider = Slider(start=0, end=10, value=5, step=0.1,
                                title="Verical cross-section position")
    rotation_slider = Slider(start=0, end=90, value=5, step=1,
                             title="Cross-section rotation")
    widgets = widgetbox(cross_vert_slider, cross_horiz_slider,
                        rotation_slider, width=250)

    fig = layout([[geomap, cross_vert], [cross_horiz, widgets]])

    output_file("seis_viewer.html")
    if showfig:
        show(fig)
    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive seismicity viewer")
    parser.add_argument(
        'filename', type=str,
        help='File containing obspy-parsable location information')
    args = parser.parse_args()
    seis_viewer(filename=args.filename)
