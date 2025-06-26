"""all function to visualize networks"""

import datetime

import numpy as np

try:
    import cartopy.crs as ccrs
    import geopandas
    import geoviews as gv
    import holoviews as hv
    import panel as pn
except ImportError as err:
    print("to use visualisation, please install libraries :")
    list_imports = ["cartopy", "geopandas", "holoviews", "panel", "geoviews"]
    print("\n".join(f"- {x}" for x in list_imports))
    raise err

from holoviews.operation.datashader import regrid
from py_eddy_tracker.generic import flatten_line_matrix
from shapely.geometry import Polygon

proj = ccrs.PlateCarree()

gv.opts.defaults(
    gv.opts.Image(colorbar=True, cmap="Spectral_r", tools=["hover"], responsive=True),
    gv.opts.Overlay(active_tools=["wheel_zoom"]),
    gv.opts.Polygons(alpha=0.8, fill_alpha=0.0, tools=["hover"]),
)


def explore(
    cyclonic=None,
    anticyclonic=None,
    hover_info=["segment", "track"],
    tiles=None,
    background_func=None,
    dynamic=False,
    date_widget="calendar",
):
    """explore NetworkObservations of cyclonic and/or anticyclonic

    :param cyclonic: cyclonic dataset, defaults to None
    :type cyclonic: :py:class:`~py_eddy_tracker.observations.network.NetworkObservations`, optional
    :param anticyclonic: anticyclonic dataset, defaults to None
    :type anticyclonic: :py:class:`~py_eddy_tracker.observations.network.NetworkObservations`, optional
    :param hover_info: list of data to show on hover, defaults to ["segment", "track"]
    :type hover_info: list, optional
    :param tiles: if specified, plot data over web tiles. like :py:class:`geoviews.source_tiles.EsriImagery`, defaults to None
    :type tiles: :py:class:`geoviews.element.geo.WMTS`, optional
    :param background_func: python function, takes as param a date and return a `xr.Dataset`, defaults to None
    :type background_func: callable, optional
    :param dynamic: if True, use :py:class:`~holoviews.operation.datashader.regrid` to delay plot of `background_func`, defaults to False
    :type dynamic: bool, optional
    :param date_widget: widget for date. if not calendar, it will be a radio, defaults to "calendar"
    :type date_widget: str, optional
    :return: bokeh plot


    .. note::

        the param `background_func` should return a :py:class :`~xarray.Dataset` with variables "lon", "lat"
        and "Grid_0001" for the data.
        The name of data should be on attribute "long_name" of Grid_0001, and "units" if wanted

       example of function :

        .. code-block:: python

            def date2dataset(date):

                fichier = f"/tmp/global_{date.strftime('%Y%m%d')}.nc"

                with xr.open_dataset(fichier) as ds:
                    ds = ds.rename({"longitude": "lon", "latitude": "lat", "adt": "Grid_0001"})
                    ds.Grid_0001.attrs["long_name"] = "adt
                    ds['lon'] = np.where(ds.lon > 180, ds.lon - 360, ds.lon)
                    ds = ds.sortby("lon").load()

                return ds
    """

    if cyclonic is None and anticyclonic is None:
        raise NotImplementedError("please specify cyclonic or anticyclonic")

    # if cyclonic or anticyclonic is not declared, only use one
    datasets = [
        (d, color)
        for d, color in [(cyclonic, "deepskyblue"), (anticyclonic, "red")]
        if d is not None
    ]

    # compute every date with data
    datasets_unique = [np.unique(d.time) for d, _ in datasets]
    unique = np.unique(np.concatenate(datasets_unique))

    if date_widget == "calendar":
        now = datetime.date(1950, 1, 1) + datetime.timedelta(int(unique[0]))
        pn_date = pn.widgets.DatePicker(
            name="date",
            value=now,
            width=200,
            enabled_dates=(
                np.datetime64("1950-01-01") + unique.astype("timedelta64[D]")
            ).tolist(),
        )
        widget_date = pn_date
    else:

        pn_date = pn.widgets.DiscretePlayer(
            name="date", options=unique.tolist(), value=unique[0]
        )

        @pn.depends(pn_date_value=pn_date.param.value)
        def plot_date(pn_date_value):
            return pn.pane.HTML(
                f"<h1>date = {np.datetime64('1950-01-01') + np.timedelta64(pn_date_value)}</h1>"
            )

        widget_date = pn.Row(pn_date, plot_date)

    @pn.depends(pn_date_value=pn_date.param.value)
    def update_all(pn_date_value):
        if isinstance(pn_date_value, datetime.date):
            # si on a choisi le widget DatePicker
            pn_date_value = (pn_date_value - datetime.date(1950, 1, 1)).days

        hv_polygons = []
        hv_exterieur = []

        for (d, color), d_unique in zip(datasets, datasets_unique):

            mask = d.time == pn_date_value
            sub_lon = ((d.contour_lon_s[mask] + 180) % 360) - 180
            sub_lat = d.contour_lat_s[mask]

            sub_data = np.moveaxis(np.array([sub_lon, sub_lat]), [0], [2])
            polygons = [Polygon(poly) for poly in sub_data]

            dct_data = {name: d[name][mask] for name in hover_info}
            dct_data["geometry"] = polygons
            gpd = geopandas.GeoDataFrame(dct_data)

            opts_common = dict(responsive=True)
            hv_polygons.append(
                gv.Polygons(gpd, vdims=hover_info).opts(line_color=color, **opts_common)
            )

            sub_lon = ((d.contour_lon_e[mask] + 180) % 360) - 180
            sub_lat = d.contour_lat_e[mask]

            _lon = flatten_line_matrix(sub_lon)
            _lat = flatten_line_matrix(sub_lat)
            hv_exterieur.append(
                gv.Path([np.array([_lon, _lat]).T]).opts(
                    color=color, alpha=0.70, line_dash="dashed", **opts_common
                )
            )

        return gv.Overlay([*hv_polygons, *hv_exterieur]).opts(
            responsive=True, tools=["hover"]
        )

    if background_func is not None:

        @pn.depends(pn_date_value=pn_date.param.value)
        def update_fond(pn_date_value):

            if isinstance(pn_date_value, datetime.date):
                # si on a choisi le widget DatePicker
                date = pn_date_value
            else:
                date = datetime.datetime(1950, 1, 1) + datetime.timedelta(
                    days=int(pn_date_value)
                )
            ds = background_func(date)

            info = ds.Grid_0001.units if hasattr(ds.Grid_0001, "units") else ""

            return gv.Image(
                (ds.lon, ds.lat, ds.Grid_0001.T),
                kdims=["lon", "lat"],
                vdims=[ds.Grid_0001.long_name],
            ).opts(responsive=True, tools=["hover"], clabel=info)

        if dynamic:
            fond = regrid(hv.DynamicMap(update_fond)).opts(height=500, responsive=True)
        else:
            fond = hv.DynamicMap(update_fond).opts(responsive=True)

        if tiles is not None:
            visu = gv.Overlay(
                [
                    tiles,
                    fond,
                    gv.DynamicMap(update_all),
                ]
            ).collate()
        else:
            visu = gv.Overlay(
                [
                    fond,
                    gv.feature.land(scale="50m").opts(
                        responsive=True, line_color="black", fill_color="darkgray"
                    ),  # , data_aspect=0.7),
                    gv.DynamicMap(update_all),
                ]
            ).collate()

        return pn.Column(
            widget_date, visu, sizing_mode="stretch_both", min_height=600, min_width=400
        )

    else:
        if tiles is not None:
            visu = gv.Overlay([tiles, gv.DynamicMap(update_all)]).collate()
        else:
            visu = gv.Overlay(
                [
                    gv.feature.land(scale="50m").opts(
                        responsive=True,
                        line_color="black",
                        fill_color="darkgray",
                        data_aspect=1,
                    ),
                    gv.DynamicMap(update_all),
                ]
            ).collate()

    return pn.Column(
        widget_date, visu, sizing_mode="stretch_both", min_height=600, min_width=400
    )
