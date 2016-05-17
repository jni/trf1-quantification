import numpy as np
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from skimage import filters, measure
import toolz as tz

def threshold(im):
    return im > filters.threshold_otsu(im)


def trf_quantify(im):
    """Quantify the TRF1 blobs in an image.

    Parameters
    ----------
    im : array of shape (M, N[, P], 3)
        The input image. The red (0th) channel should contain TRF1
        signal, while the blue (2nd) channel should contain chromatin
        signal.

    Returns
    -------
    props : list of list of float
        The desired properties measured for each blob in the image:
        blob size, raw mean intensity, raw total intensity, raw max
        intensity, 
        and eccentricity.
    """
    trf = im[..., 0]
    objs = nd.label(threshold(trf))[0]
    trfprops = measure.regionprops(objs, trf)
    # non-intensity features
    sizes = [p.area for p in trfprops]
    eccen = [p.eccentricity for p in trfprops]
    # unnormalised properties (raw)
    rmean = [p.mean_intensity for p in trfprops]
    rtotl = [s * m for s, m in zip(sizes, rmean)]
    rmaxs = [p.max_intensity for p in trfprops]
    # post-normalised properties
    chrom = im[..., 2]
    chrprops = measure.regionprops(objs, chrom)
    means = [p.mean_intensity / q.mean_intensity
             for p, q in zip(trfprops, chrprops)]
    total = [s * m for s, m in zip(sizes, means)]
    maxes = [p.max_intensity / q.max_intensity
             for p, q in zip(trfprops, chrprops)]
    # pre-normalised properties
    trf = trf.astype(float) / (chrom + 1)
    trfprops = measure.regionprops(objs, trf)
    nmean = [p.mean_intensity for p in trfprops]
    ntotl = [s * m for s, m in zip(sizes, rmean)]
    nmaxs = [p.max_intensity for p in trfprops]
    return list(zip(sizes, rmean, rtotl, rmaxs,
                           means, total, maxes,
                           nmean, ntotl, nmaxs, eccen))


def boxplot(im_nums, kd, values):
    """Show a boxplot with samples grouped by `im_nums` and coloured by `kd`.

    All three input lists should have the same length.

    Parameters
    ----------
    im_nums : list of int
        The image number for this blob.
    kd : list of string
        The status of a blob as either knockdown or control.
    values : list of float
        The actual values to be plotted.

    Returns
    -------
    fig : Pyplot figure
        The boxplot reference
    """
    palette = ['blue', 'orange', 'darkgreen', 'purple']
    fig = plt.figure(figsize=(12, 3))
    values_by_image = {}
    for im_num, val in zip(im_nums, values):
        values_by_image.setdefault(im_num, []).append(val)
    image_kind = {im : k for im, k in zip(im_nums, kd)}
    x_vals = np.unique(im_nums)
    kinds = sorted(list(set(kd)))
    for c, k in zip(palette, kinds):
        to_plot = [[]] * len(x_vals)
        for i in x_vals:
            if image_kind[i] == k:
                to_plot[i] = values_by_image[i]
        _ = plt.boxplot(to_plot, labels=x_vals, boxprops={'color': c})
    return fig


def scatter(kd, control, colors=['orange', 'blue'], **kwargs):
    """Show a jittered scatterplot of the measurements.

    Parameters
    ----------
    kd : list of list of float
        The list of `trf_quantify` results for all AUKB knockdown
        images in the dataset. (Each result is itself a list.)
    control : list of list of float
        The list of `trf_quantify` results for all control images in
        the dataset.
    colors : list of two matplotlib colorspecs, optional
        The colors corresponding to AUKB-KD (0) and control (1) data
        points on the scatterplot.
    
    Additional Parameters
    ---------------------
    **kwargs : keyword arguments
        Additional keyword arguments passed directly to
        ``plt.scatter``.

    Returns
    -------
    fig : matplotlib axes
        The returned value from the call to ``plt.scatter``.
    """
    xs = list(tz.concat([i + 0.2 * np.random.randn(n)
                         for i, n in enumerate(map(len, kd + control))]))
    color_vector = ([colors[0]] * sum(map(len, kd)) +
                    [colors[1]] * sum(map(len, control)))
    ys = list(tz.concat(kd + control))
    fig = plt.scatter(xs, ys, c=color_vector, **kwargs)
    plt.xlim(0, max(xs) + 1)
    plt.ylim(0, max(ys) + 1)
    return fig
