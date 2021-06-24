__all__ = [
    "align",
    "threshold",
    "prune_labels",
    "prune_by_area",
    "clicker",
    "point_associater"
]
from skimage import transform
from skimage.segmentation import relabel_sequential
from skimage.registration import optical_flow_ilk, optical_flow_tvl1
import matplotlib.pyplot as plt
import numpy as np
import copy
import mpl_interactions.ipyplot as iplt


def align_optical_flow(BF, RM, flip_rm=True, plot=False):
    """
    Parameters
    ----------
    BF : (M, N) array-like
    RM : (Y, X) array-like
        If a different shape than BF it will be resized
    flip_rm : bool, default: True
        Whether to flip the RM image.
    plot : bool, default: True
        Whether to generate a plot of the fit

    Returns
    -------
    u, v : (M, N) array
        The shifts
    warped : (M, N) array
        The raman image warped to match the BF
    make_plot : function
        A function to

    """
    rm_resized = transform.resize(RM, BF.shape)
    if flip_rm:
        rm_resized = np.flip(rm_resized)
    v, u = optical_flow_tvl1(BF, rm_resized)
    nr, nc = BF.shape

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    warped = transform.warp(
        rm_resized, np.array([row_coords + v, col_coords + u]), mode="nearest"
    )

    def make_plot(axs=None):
        """
        Parameters
        ----------
        axs : (3,) matplotlib axes
        """
        if axs is None:
            fig, axs = plt.subplots(1, 3)

        rm_cmap = "Reds"
        over_alpha = 0.5
        axs[0].imshow(BF, cmap="gray")
        axs[0].imshow(rm_resized, alpha=over_alpha, cmap=rm_cmap)
        axs[0].set_title("Just linear scaling")

        axs[1].imshow(BF, cmap="gray")
        axs[1].imshow(warped, alpha=over_alpha, cmap=rm_cmap)
        axs[1].set_title("optical flow")

        nvec = 20  # Number of vectors to be displayed along each image dimension
        nl, nc = BF.shape
        step = max(nl // nvec, nc // nvec)

        y, x = np.mgrid[:nl:step, :nc:step]
        u_ = u[::step, ::step]
        v_ = v[::step, ::step]

        # ax1.imshow(norm)

        axs[2].quiver(
            x, y, u_, v_, color="r", units="dots", angles="xy", scale_units="xy", lw=3
        )
        axs[2].set_title("Optical flow magnitude and vector field")

    if plot:
        make_plot()

    return u, v, warped, make_plot


def threshold(image, nsteps=100, cmap=None):
    """
    Set up an interactive thresholding

    Parameters
    ----------
    image : (M, N) array-like
    nsteps : int, default: 200
        The number of steps in vmin/vmax
    cmap : matplotlib colormap, optional
        Defaults to a clipped viridis

    Returns
    -------
    controls
    """
    if cmap is None:
        cmap = copy.copy(plt.cm.viridis)
        cmap.set_under(alpha=0)
    im = np.asarray(image)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # plot histogram of pixel intensities
    axs[1].hist(im.flatten(), bins="auto")
    axs[1].set_title("Histogram of Pixel Intensities")

    # create interactive controls
    ctrls = iplt.imshow(
        im, vmin_vmax=("r", im.min(), im.max(), nsteps), ax=axs[0], cmap=cmap
    )
    iplt.axvline(ctrls["vmin"], ax=axs[1], c="k")
    _ = iplt.axvline(ctrls["vmax"], ax=axs[1], c="k")
    return ctrls


def prune_labels(labels, remove, inplace=True):
    """
    Parameters
    ----------
    labels : (M,N) arraylike of int
    remove : array-like of int
        The set of labels to remove from the array
    inplace : bool, default: True
        Whether to modify the array in place

    Returns
    -------
    pruned : (M, N) array-like of int
        The labels array with the specified labels removed
    """
    if inplace:
        out = labels
    else:
        out = np.copy(labels)

    for i in remove:
        out[labels == i] = 0

    return relabel_sequential(out)[0]


def prune_by_area(labels, min_area=0, max_area=None, inplace=True):
    """
    Parameters
    ----------
    labels : (M, N) arraylike of int
    min_area : int, default: 0
    max_area : int, optional
    inplace : bool, default: True
        Whether to modify the array in place

    Returns
    -------
    pruned : (M, N) array-like of int
        The labels array with the specified labels removed
    """
    ids, areas = np.unique(labels, return_counts=True)
    idx = areas > min_area
    if max_area is not None:
        idx *= areas < max_area
    return prune_labels(labels, ids[~idx])


# def prune_by_regionprop(labels, prop, min=None, max=None):
#     """
#     Calculate the regionprop
#     """


def click_location_tester(BF, RM, bf_to_rm, scat_kw=None):
    """
    Parameters
    ----------
    BF, RM : (M, N) array-like
    bf_to_rm : Callable
        Function to convrt from BF pixel coordinates to RM pixel coordinates
    scat_kw : dict
        kwargs for the scatters
    """
    if scat_kw is None:
        scat_kw = {"c": "r"}
    import ipywidgets as widgets

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(BF)
    axs[1].imshow(RM)
    axs[0].set_title("BF")
    axs[1].set_title("RM")
    scats = []
    scats.append(axs[0].scatter([], [], **scat_kw))
    scats.append(axs[1].scatter([], [], **scat_kw))

    def on_click(event):
        if fig.canvas.widgetlock.locked():
            return
        if event.inaxes is axs[0]:
            data = scats[0].get_offsets()
            data = np.ma.append(data, np.asarray([[event.xdata, event.ydata]]), axis=0)
            scats[0].set_offsets(data)
            scats[1].set_offsets(bf_to_rm(data))
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

from scipy.spatial.distance import cdist
from mpl_interactions import panhandler, zoom_factory
from matplotlib.backend_bases import MouseButton
class clicker:
    def __init__(self, ax1, ax2, forward):  # , backward):
        self.ax1 = ax1
        self.ax2 = ax2
        self.forward = forward
        #         self.backward = backward
        self.ax1.figure.canvas.mpl_connect("button_press_event", self._forward)
        self.ax2.figure.canvas.mpl_connect("button_press_event", self._backward)
        self._positions_1 = []
        self._positions_2 = []
        self._scatter1 = ax1.scatter([], [])
        self._scatter2 = ax2.scatter([], [])
        self._blarg = ax2.scatter([], [])
        self._figs = set([ax1.figure, ax2.figure])

    def _forward(self, event):
        if not self.ax1.figure.canvas.widgetlock.available(self):
            return
        if event.inaxes is self.ax1:
            if event.button is MouseButton.LEFT:
                # swapping x and y because imshow transposes
                self._positions_1.append((event.ydata, event.xdata))
                self._positions_2.append(np.squeeze(self.forward(*self._positions_1[-1])))
                self._update_points()
            elif event.button is MouseButton.RIGHT:
                if len(self._positions_1) == 0:
                    return
                dists = cdist([[event.ydata, event.xdata]], self._positions_1)
                idx = np.argmin(dists[0])
                self._positions_1.pop(idx)
                self._positions_2.pop(idx)
                self._update_points()


    def _update_points(self):
        # swapping x and y because imshow transposes
        self._scatter1.set_offsets(np.asarray(self._positions_1)[:, ::-1])
        self._scatter2.set_offsets(np.asarray(self._positions_2)[:, ::-1])
        for fig in self._figs:
            fig.canvas.draw()

    def _backward(self, event):
        pass

    @property
    def positions1(self):
        return np.asarray(self._positions_1)

    @property
    def positions2(self):
        return np.asarray(self._positions_2)

    @positions1.setter
    def positions1(self, value):
        value = np.asarray(value)
        if value.ndim != 2:
            raise ValueError("Positions must be 2D with shape (N, 2)")
        self._positions_1 = value.tolist()

    @positions2.setter
    def positions2(self, value):
        value = np.asarray(value)
        if value.ndim != 2:
            raise ValueError("Positions must be 2D with shape (N, 2)")
        self._positions_2 = value.tolist()


class point_associater(clicker):
    def __init__(self, im1, im2, forward, axs=None, fig_kws={}):  # , backward):
        if axs is None:
            fig, axs = plt.subplots(1, 2, **fig_kws)

        axs[0].imshow(im1)
        axs[1].imshow(im2)

        self._phs = []
        for fig in set([ax.figure for ax in axs]):
            self._phs.append(panhandler(fig, button=MouseButton.MIDDLE))
        zoom_factory(axs[0])
        zoom_factory(axs[1])

        super().__init__(axs[0], axs[1], forward)

        def bf_ylim(event_ax):
            X = event_ax.get_ylim(), event_ax.get_xlim()
            # swapping x and y because imshow transposes
            new_lims = forward(event_ax.get_ylim(), event_ax.get_xlim())
            axs[1].set_xlim(new_lims[:, 1])
            axs[1].set_ylim(new_lims[:, 0])

        # have to use `y` so that we get both changes
        # seems as though the x callback can fire before the y limits have been changed
        axs[0].callbacks.connect("ylim_changed", bf_ylim)


