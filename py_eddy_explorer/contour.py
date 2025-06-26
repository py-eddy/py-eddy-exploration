import datetime

import holoviews as hv
from holoviews.operation.datashader import ResampleOperation2D as ResampleOperation
import numba
import numpy as np
import pandas
import panel
import param
from py_eddy_tracker import generic


@numba.njit(cache=True)
def lines_clip_by_box(x00, y00, x11, y11, x2, y2, x3, y3):  # pragma: no cover
    xref, yref = x00, y00
    x00, x11, x2, x3 = 0.0, x11 - xref, x2 - xref, x3 - xref
    y00, y11, y2, y3 = 0.0, y11 - yref, y2 - yref, y3 - yref

    for x0, y0, x1, y1 in (
        (x00, y00, x00, y11),
        (x00, y11, x11, y11),
        (x11, y11, x11, y00),
        (x11, y00, x00, y00),
    ):
        dx01 = x0 - x1
        dy01 = y0 - y1
        dx23 = x2 - x3
        dy23 = y2 - y3

        # nulle si //
        c = dx01 * dy23 - dx23 * dy01
        if abs(c) < 1e-9:
            continue
        a = x0 * y1 - x1 * y0
        b = x2 * y3 - x3 * y2
        x = (a * dx23 - b * dx01) / c
        y = (a * dy23 - b * dy01) / c
        if y0 == y1:
            x0_ = max(min(x2, x3), min(x0, x1))
            x1_ = min(max(x2, x3), max(x0, x1))
            if x0_ == x1_:
                if y > max(y2, y3) or y < min(y2, y3):
                    continue
            elif x < x0_ or x > x1_:
                continue
        if x0 == x1:
            y0_ = max(min(y2, y3), min(y0, y1))
            y1_ = min(max(y2, y3), max(y0, y1))
            if y0_ == y1_:
                if x > max(x2, x3) or x < min(x2, x3):
                    continue
            elif y < y0_ or y > y1_:
                continue
        return x + xref, y + yref
    return np.nan, np.nan


@numba.njit(cache=True)
def is_in_box(x, y, x0, x1, y0, y1):  # pragma: no cover
    return (x > x0) & (x < x1) & (y > y0) & (y < y1)


@numba.njit(cache=True)
def xy_in_box(x, y, x0, x1, y0, y1):  # pragma: no cover
    x0, x1, y0, y1 = x0, x1, y0, y1
    nb = x.size
    # Create bigger than need ...
    x_new, y_new = np.empty(nb * 2 + 50, dtype=x.dtype), np.empty(
        nb * 2 + 50, dtype=y.dtype
    )
    previous_in_box = True
    x_previous, y_previous = x[0], y[0]
    i_ = 0
    for i in range(nb):
        x_, y_ = x[i], y[i]
        current_in_box = is_in_box(x_, y_, x0, x1, y0, y1)
        if current_in_box:
            if not previous_in_box and not np.isnan(x_previous):
                x_new[i_], y_new[i_] = lines_clip_by_box(
                    x0, y0, x1, y1, x_previous, y_previous, x_, y_
                )
                i_ += 1
            x_new[i_] = x_
            y_new[i_] = y_
            previous_in_box = True
            i_ += 1
        else:
            if previous_in_box:
                if not np.isnan(x_):
                    x_new[i_], y_new[i_] = lines_clip_by_box(
                        x0, y0, x1, y1, x_previous, y_previous, x_, y_
                    )
                    i_ += 1
                    x_new[i_] = np.nan
                    y_new[i_] = np.nan
                else:
                    x_new[i_] = np.nan
                    y_new[i_] = np.nan
                i_ += 1
            previous_in_box = False

        previous_in_box = current_in_box
        x_previous, y_previous = x_, y_
    return x_new[:i_], y_new[:i_]


class GeoResample(ResampleOperation):
    def ax_properties(self):
        x0, x1 = xrange = self.p.x_range if self.p.x_range else (0, 360)
        x1_ = min(x0 + 360, x1)
        xrange = np.float64(x0), np.float64(x1_)
        if xrange[1] <= xrange[0]:
            xrange = (xrange[1], xrange[0])
        yrange = tuple(np.float64(self.p.y_range if self.p.y_range else (-90, 90)))

        if yrange[1] <= yrange[0]:
            yrange = (yrange[1], yrange[0])

        pixel_ratio = self._get_pixel_ratio()
        width = int(self.p.width / pixel_ratio)
        height = int(self.p.height / pixel_ratio)
        width = int(width * (360 / (x1 - x0) if x1 != x1_ else 1))
        return xrange, yrange, x1, width, height

    def _store_params_opts(self, kdims, vdims, transform, params=dict(), opts=dict()):
        self.kdims = kdims
        self.vdims = vdims
        self.transform = transform
        self._stored_params = dict(**params)
        self._computed_opts = dict(**opts)

    def _build_element(self, obj, extra_params=dict(), extra_opts=dict()):
        "build element depending if geo or not."

        params = dict(**self._stored_params)
        params.update(extra_params)
        opts = dict(**self._computed_opts)
        opts.update(extra_opts)

        img_class = getattr(hv, self.PLOTTING_TYPE)

        vdims = [self.transform.get(vdim, vdim) for vdim in self.vdims]
        kdims = [self.transform.get(kdim, kdim) for kdim in self.kdims]

        element = img_class(obj, kdims=kdims, vdims=vdims, **params).options(**opts)
        return element

    def _process(self, element, key=None):
        hv_element = self._compute_hv_element()
        return hv_element

    def _compute_hv_element(self):
        xrange, yrange, x1, width, height = self.ax_properties()
        try:
            ds = self.compute(xrange, yrange, x1, width, height)
        except Exception as err:
            import traceback

            print(traceback.format_exc())
            raise err

        opts = ds.attrs.get("hv_opts", dict())

        element_agg = self._build_element(ds, extra_opts=opts)
        element_agg.extents = (xrange[0], yrange[0], xrange[1], yrange[1])
        return element_agg


class EddyContour(GeoResample):
    time = param.Date(
        None,
        bounds=(datetime.datetime(1992, 1, 1), datetime.datetime(2050, 1, 1)),
        doc="Allow date sliding",
    )
    delta = param.Number(
        0.5, bounds=(0, None), doc="Number of day around reference time"
    )
    PLOTTING_TYPE = "Path"

    def __call__(self, ds, **kwargs):
        self.ds = ds
        self._store_params_opts(
            ["longitude", "latitude"], [], {}, opts=dict(data_aspect=1, color="k", line_width=0.5)
        )
        return super().__call__(hv.Element([]), **kwargs)

    def contour(self, x0, x1, y0, y1):
        r = (y1 - y0) / self.p.height
        t = (np.datetime64(self.p.time) - np.datetime64("1950-01-01")) / np.timedelta64(
            1, "D"
        )
        if isinstance(self.p.delta, datetime.timedelta) or isinstance(
            self.p.delta, np.timedelta64
        ):
            delta = int(np.timedelta64(self.p.delta) / np.timedelta64(1, "D"))
        else:
            delta = int(self.p.delta)
        i = self.ds.daily_time_indexer(t, delta)
        xe = generic.flatten_line_matrix(self.ds.contour_lon_e[i])
        ye = generic.flatten_line_matrix(self.ds.contour_lat_e[i])
        xe, ye = generic.simplify(xe, ye, precision=max(r, 0.001))
        xe, ye = generic.wrap_longitude(xe, ye, x0, cut=True)
        return xy_in_box(xe, ye, x0, x1, y0, y1)

    def compute(self, xrange, yrange, x1, width, height):
        x0, x1 = self.p.x_range if self.p.x_range else (0.0, 360.0)
        y0, y1 = self.p.y_range if self.p.y_range else (-90.0, 90.0)

        xdim, ydim = self.kdims

        x, y = self.contour(x0, x1, y0, y1)
        lines = pandas.DataFrame({xdim: x.astype("f4"), ydim: y.astype("f4")})
        return lines

    @classmethod
    def widgets(
        cls,
        ds,
        vertical=True,
        width=400,
        default_time=-1,
        default_delta=0.5,
        player=False,
    ):
        t = (
            np.arange(*ds.period).astype("timedelta64[D]") + np.datetime64("1950-01-01")
        ).astype(datetime.datetime)

        widget = (
            panel.widgets.DiscretePlayer if player else panel.widgets.DiscreteSlider
        )
        d_time = t[default_time] if isinstance(default_time, int) else default_time
        d_delta = default_delta * np.timedelta64(1, "D")

        time_selector = widget(
            value=d_time, options={str(v): v for v in t}, width=width
        )
        delta_selector = panel.widgets.FloatSlider(
            value=d_delta,
            start=0.1,
            end=5.0,
            step=0.1,
            name="Delta time in days",
            width=width,
        )
        delta_selector.value = d_delta
        layout = panel.Column if vertical else panel.Row
        return layout(time_selector, delta_selector), dict(
            time=time_selector.param.value, delta=delta_selector.param.value
        )
