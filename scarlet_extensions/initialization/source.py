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
        considered to have edge flux
    Returns
    -------
    isEdge: `bool`
        Whether or not the source has flux on the edge.
    """
    assert edgeDistance > 0

    # Use the first band that has a non-zero SED
    band = np.min(np.where(source.sed > 0)[0])
    model = source.get_model()[band]
    for edge in range(edgeDistance):
        if (
            np.any(model[edge-1] > 0) or
            np.any(model[-edge] > 0) or
            np.any(model[:, edge-1] > 0) or
            np.any(model[:, -edge] > 0)
        ):
            return True
    return False


def initSource(frame, center, observation,
               symmetric=False, monotonic=True,
               thresh=5, components=1, edgeDistance=1, shifting=False):
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
    peak : `PeakRecord`
        Record for a peak in the parent `PeakCatalog`
    observation : `LsstObservation`
        The images, psfs, etc, of the observed data.
    bbox : `lsst.geom.Box2I`
        The bounding box of the parent footprint.
    symmetric : `bool`
        Whether or not the object is symmetric
    monotonic : `bool`
        Whether or not the object has flux monotonically
        decreasing from its center
    thresh : `float`
        Fraction of the background to use as a threshold for
        each pixel in the initialization
    components : int
        The number of components for the source.
        If `components=0` then a `PointSource` model is used.
    """
    while components > 1:
        try:
            source = MultiComponentSource(frame, center, observation, symmetric=symmetric,
                                          monotonic=monotonic, thresh=thresh, shifting=shifting)
            if (np.any([np.any(np.isnan(c.sed)) for c in source.components]) or
                    np.any([np.all(c.sed <= 0) for c in source.components])):
                logger.warning("Could not initialize")
                raise ValueError("Could not initialize source")
            if hasEdgeFlux(source, edgeDistance):
                source.shifting = True
            break
        except Exception:
            # If the MultiComponentSource failed to initialize
            # try an ExtendedSource
            components -= 1

    if components == 1:
        try:
            source = ExtendedSource(frame, center, observation, thresh=thresh,
                                    symmetric=symmetric, monotonic=monotonic, shifting=shifting)
            if np.any(np.isnan(source.sed)) or np.all(source.sed <= 0) or np.sum(source.morph) == 0:
                logger.warning("Could not initialize")
                raise ValueError("Could not initialize source")
        except Exception:
            # If the source is too faint for background detection,
            # initialize it as a PointSource
            components -= 1

    if components == 0:
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
                              symmetric, monotonic, thresh, components,
                              edgeDistance, shifting=True)
        source.isEdge = True
    else:
        source.isEdge = False

    return source
