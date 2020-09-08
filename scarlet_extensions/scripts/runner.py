import numpy as np
import scarlet.display
from scarlet.initialization import build_initialization_coadd
from ..initialization.detection import makeCatalog


class Runner:
    """ Class that lets scarlet run on a set of `Data` objects

        This class makes running on batches of images easier.
        It runs on multiple or single observations alike.

        parameters
        ----------
        datas: list of `scarlet_extensions.initialization.detection.Data` objects
            the list of Data objects to run scarlet on.
            These objects contain a cube of images along with psf, wcs anc channels information
        model_psf: 'numpy.ndarray' or `scarlet.PSF`
            the target psf of the model
        ra_dec: `array`
            ra and dec positions of detected sources
    """

    def __init__(self, data, model_psf, ra_dec = None):

        self._data = data
        self.run_detection(lvl = 3, wavelet = True)

        if len(self._data) == 1:
            weight = np.ones_like(self._data[0].images) / (self.bg_rms ** 2)[:, None, None]
            observations = [scarlet.Observation(self._data[0].images,
                                                    wcs=self._data[0].wcs,
                                                    psfs=self._data[0].psfs,
                                                    channels=self._data[0].channels,
                                                    weights=weight)]
        else:
            observations = []
            for i,bg in enumerate(self.bg_rms):
                weight = np.ones_like(self._data[i].images) / (bg**2)[:,None,None]
                observations.append(scarlet.Observation(self._data[i].images,
                                                    wcs=self._data[i].wcs,
                                                    psfs=self._data[i].psfs,
                                                    channels=self._data[i].channels,
                                                    weights=weight))
        self.observations = observations
        self.frame = scarlet.Frame.from_observations(self.observations, model_psf, coverage = 'intersection')
        # Convert the HST coordinates to the HSC WCS
        loc = [type(o) is not scarlet.LowResObservation for o in self.observations]
        if ra_dec is None:
            self.ra_dec = self.observations[np.where(loc)[0][0]].frame.get_sky_coord(self.pixel_coords)
        else:
            self.ra_dec = ra_dec

    def run(self, it = 200, e_rel = 1.e-6, plot = False):
        """ Run scarlet on the Runner object

        parameters
        ----------
        it: `int`
            Maximum number of iterations used to fit the model
        e_rel: `float`
            limit on the convergence: stop condition
        plot: 'bool'
            if set to True, plots the model and residuals of the regression
        """
        self.blend = scarlet.Blend(self.sources, self.observations)
        self.blend.fit(it, e_rel=e_rel)
        print("scarlet ran for {0} iterations to logL = {1}".format(len(self.blend.loss), -self.blend.loss[-1]))
        if plot:
            from scarlet.display import AsinhMapping
            norm_hsc = AsinhMapping(minimum=-1, stretch=5, Q=1)
            norm_hst = AsinhMapping(minimum=-1, stretch=10, Q=1)
            norms = [norm_hsc, norm_hst]
            import matplotlib.pyplot as plt
            for i in range(len(self.observations)):
                scarlet.display.show_scene(self.sources,
                                       norm=norms[i],
                                       observation=self.observations[i],
                                       show_model=False,
                                       show_rendered=True,
                                       show_observed=True,
                                       show_residual=True,
                                       figsize=(12, 4)
                                       )
            plt.show()

    def initialize_sources(self, ks, ra_dec = None):
        '''
        Initialize all sources as Extended sources
        ks: array
            array of sources for the scene. For elements of ks that are numbers, the source id an extended source.
            If an element of ks is set to 'point', it corresponds to a point source.
        '''
        if ra_dec is not None:
            self.ra_dec = ra_dec
        # Building a detection coadd
        coadd, bg_cutoff = build_initialization_coadd(self.observations, filtered_coadd=True)


        # Source initialisation
        sources = []
        for i, sky in enumerate(self.ra_dec):
            if ks[i] == 'point':
                sources.append(
                    scarlet.PointSource(self.frame, sky, self.observations))
            else:
                sources.append(
                    scarlet.ExtendedSource(self.frame, sky, self.observations, coadd=coadd, coadd_rms=bg_cutoff))
        self.sources = sources

    def run_detection(self, lvl = 3, wavelet = True):
        ''' Runs the detection algorithms on data

        Parameters
        ----------
        lvl: float
            Detection level in units of background noise
        wavelet: Bool
            if set to true, runs sep on a wavelet filtered image

        Returns
        -------
        catalog: dict
            detection catalog that contains at least the position of the detected sources
        bg_rms: float
            background root mean square of the image
        '''
        self.lvl = lvl
        self.wavelet = wavelet
        catalog, self.bg_rms = makeCatalog(self._data, lvl, wavelet)
        # Get the source coordinates from the HST catalog
        self.pixel_coords = np.stack((catalog['y'], catalog['x']), axis=1)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.run_detection(self.lvl, self.wavelet)
        for i,obs in enumerate(self.observations):
            obs.images = self._data[i].images
        loc = [type(o) is not scarlet.LowResObservation for o in self.observations]
        self.ra_dec = self.observations[np.where(loc)[0][0]].frame.get_sky_coord(self.pixel_coords)
