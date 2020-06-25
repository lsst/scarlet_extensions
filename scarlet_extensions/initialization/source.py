import logging

import numpy as np
from scarlet.source import PointSource, ExtendedSource, MultiComponentSource


logger = logging.getLogger("scarlet_extensions.initialization.source")


def hasEdgeFlux(source, edgeDistance=1):
    """hasEdgeFlux

    Determine whether or not a source has flux within `edgeDistance`
    of the edge.

    Parameters
    ----------
    source : `scarlet.Component`
        The source to check for edge flux
    edgeDistance : int
        The distance from the edge of the image to consider
        a source an edge source. For example if `edgeDistance=3`
        then any source within 3 pixels of the edge will be
        considered to have edge flux.
        If `edgeDistance` is `None` then the edge check is ignored.

    Returns
    -------
    isEdge: `bool`
        Whether or not the source has flux on the edge.
    """
    if edgeDistance is None:
        return False

    assert edgeDistance > 0

    # Use the first band that has a non-zero SED
    if hasattr(source, "sed"):
        band = np.min(np.where(source.sed > 0)[0])
    else:
        band = np.min(np.where(source.components[0].sed > 0)[0])
    model = source.get_model()[band]
    for edge in range(edgeDistance):
        if (
            np.any(model[edge-1] > 0)
            or np.any(model[-edge] > 0)
            or np.any(model[:, edge-1] > 0)
            or np.any(model[:, -edge] > 0)
        ):
            return True
    return False


def initAllSources(frame, centers, observation,
                   symmetric=False, monotonic=True,
                   thresh=1, maxComponents=1, edgeDistance=1, shifting=False,
                   downgrade=True, fallback=True, minGradient=0):
    """Initialize all sources in a blend

    Any sources which cannot be initialized are returned as a `skipped`
    index, the index needed to reinsert them into a catalog to preserve
    their index in the output catalog.

    See `~initSources` for a description of the parameters

    Parameters
    ----------
    centers : list of tuples
        `(y, x)` center location for each source

    Returns
    -------
    sources: list
        List of intialized sources, where each source derives from the
        `~scarlet.Component` class.
    """
    # Only deblend sources that can be initialized
    sources = []
    skipped = []
    for k, center in enumerate(centers):
        source = initSource(
            frame, center, observation,
            symmetric, monotonic,
            thresh, maxComponents, edgeDistance, shifting,
            downgrade, fallback, minGradient)
        if source is not None:
            sources.append(source)
        else:
            skipped.append(k)
    return sources, skipped


def initSource(frame, center, observation,
               symmetric=False, monotonic=True,
               thresh=1, maxComponents=1, edgeDistance=1, shifting=False,
               downgrade=True, fallback=True, minGradient=0):
    """Initialize a Source

    The user can specify the number of desired components
    for the modeled source. If scarlet cannot initialize a
    model with the desired number of components it continues
    to attempt initialization of one fewer component until
    it finds a model that can be initialized.
    It is possible that scarlet will be unable to initialize a
    source with the desired number of components, for example
    a two component source might have degenerate components,
    a single component source might not have enough signal in
    the joint coadd (all bands combined together into
    single signal-to-noise weighted image for initialization)
    to initialize, and a true spurious detection will not have
    enough signal to initialize as a point source.
    If all of the models fail, including a `PointSource` model,
    then this source is skipped.

    Parameters
    ----------
    frame : `LsstFrame`
        The model frame for the scene
    center : `tuple` of `float``
        `(y, x)` location for the center of the source.
    observation : `~scarlet.Observation`
        The `Observation` that contains the images, weights, and PSF
        used to generate the model.
    symmetric : `bool`
        Whether or not the object is symmetric
    monotonic : `bool`
        Whether or not the object has flux monotonically
        decreasing from its center
    thresh : `float`
        Fraction of the background to use as a threshold for
        each pixel in the initialization
    maxComponents : int
        The maximum number of components in a source.
        If `fallback` is `True` then when
        a source fails to initialize with `maxComponents` it
        will continue to subtract one from the number of components
        until it reaches zero (which fits a point source).
        If a point source cannot be fit then the source is skipped.
    edgeDistance : int
        The distance from the edge of the image to consider
        a source an edge source. For example if `edgeDistance=3`
        then any source within 3 pixels of the edge will be
        considered to have edge flux.
        If `edgeDistance` is `None` then the edge check is ignored.
    shifting : bool
        Whether or not to fit the position of a source.
        This is an expensive operation and is typically only used when
        a source is on the edge of the detector.
    downgrade : bool
        Whether or not to decrease the number of components for sources
        with small bounding boxes. For example, a source with no flux
        outside of its 16x16 box is unlikely to be resolved enough
        for multiple components, so a single source can be used.
    fallback : bool
        Whether to reduce the number of components
        if the model cannot be initialized with `maxComponents`.
        This is unlikely to be used in production
        but can be useful for troubleshooting when an error can cause
        a particular source class to fail every time.
    """
    while maxComponents > 1:
        try:
            source = MultiComponentSource(frame, center, observation, symmetric=symmetric,
                                          monotonic=monotonic, thresh=thresh, shifting=shifting)
            if (np.any([np.any(np.isnan(c.sed)) for c in source.components])
                    or np.any([np.all(c.sed <= 0) for c in source.components])
                    or np.any([np.any(~np.isfinite(c.morph)) for c in source.components])):
                msg = "Could not initialize source at {} with {} components".format(center, maxComponents)
                logger.warning(msg)
                raise ValueError(msg)

            if downgrade and np.all([np.all(np.array(c.bbox.shape[1:]) <= 8) for c in source.components]):
                # the source is in a small box so it must be a point source
                maxComponents = 0
            elif downgrade and np.all([np.all(np.array(c.bbox.shape[1:]) <= 16) for c in source.components]):
                # if the source is in a slightly larger box
                # it is not big enough to model with 2 components
                maxComponents = 1
            elif hasEdgeFlux(source, edgeDistance):
                source.shifting = True

            break
        except Exception as e:
            if not fallback:
                raise e
            # If the MultiComponentSource failed to initialize
            # try an ExtendedSource
            maxComponents -= 1

    if maxComponents == 1:
        try:
            source = ExtendedSource(frame, center, observation, thresh=thresh,
                                    symmetric=symmetric, monotonic=monotonic, shifting=shifting,
                                    min_grad=minGradient)
            if np.any(np.isnan(source.sed)) or np.all(source.sed <= 0) or np.sum(source.morph) == 0:
                msg = "Could not initlialize source at {} with 1 component".format(center)
                logger.warning(msg)
                raise ValueError(msg)

            if downgrade and np.all(np.array(source.bbox.shape[1:]) <= 16):
                # the source is in a small box so it must be a point source
                maxComponents = 0
            elif hasEdgeFlux(source, edgeDistance):
                source.shifting = True
        except Exception as e:
            if not fallback:
                raise e
            # If the source is too faint for background detection,
            # initialize it as a PointSource
            maxComponents -= 1

    if maxComponents == 0:
        try:
            source = PointSource(frame, center, observation)
        except Exception:
            # None of the models worked to initialize the source,
            # so skip this source
            return None

    if hasEdgeFlux(source, edgeDistance):
        # The detection algorithm implemented in meas_algorithms
        # does not place sources within the edge mask
        # (roughly 5 pixels from the edge). This results in poor
        # deblending of the edge source, which for bright sources
        # may ruin an entire blend. So we reinitialize edge sources
        # to allow for shifting and return the result.
        if not isinstance(source, PointSource) and not shifting:
            return initSource(frame, center, observation,
                              symmetric, monotonic, thresh, maxComponents,
                              edgeDistance, shifting=True)
        source.isEdge = True
    else:
        source.isEdge = False

    return source
