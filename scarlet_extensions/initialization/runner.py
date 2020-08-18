import numpy as np
import scarlet.display
from scarlet.display import AsinhMapping
from scarlet import Starlet
from scarlet.wavelet import mad_wavelet
import scipy.stats as scs
from scarlet.initialization import build_initialization_coadd
from functools import partial


class Runner:

    def __init__(self, datas, model_psf):
        '''

        parameters
        ----------
        datas: list of Data objects
            the list of Data objects to run scarlet on
        '''
        self.data = datas
        run_detection(self, lvl = 3, wavelet = True)

        # Get the source coordinates from the HST catalog
        pixels = np.stack((self.catalog['y'], self.catalog['x']), axis=1)

        observations = []
        for bg in self.bg_rms:
            weight = np.ones_like(d.images) / (bg**2)[:,None,None]
            observations.append(scarlet.Observation(self.data.images, self.data.psfs, self.data.channels, weight))
        self.observations = observations
        self.frame = scarlet.Frame.from_observations(self.observations, model_psf, coverage = 'intersection')

        # Convert the HST coordinates to the HSC WCS
        self.ra_dec = self.observations[0].frame.get_sky_coord(pixels)

    def run(self, it = 200, e_rel = 1.e-6):
        self.blend = scarlet.Blend(self.sources, self.observations)
        self.blend.fit(it, e_rel=e_rel)
        print("scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))


    def initialise_sources(self):
        '''

        '''
        if len(self.observations) > 1:
            # Building a detection coadd
            coadd, bg_cutoff = build_initialization_coadd(self.observations, filtered_coadd=True)
        else:
            coadd = None
            bg_cutoff = None
        # Source initialisation
        self.sources = [
            scarlet.ExtendedSource(self.frame, sky, obs, coadd=coadd, coadd_rms=bg_cutoff)
            for sky in self.ra_dec
        ]



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
        self.detection = makeCatalog(self.data, lvl, wavelet)


    @data.setter
    def data(self, data):
        self.data = data
        self.run_detection(self, self.lvl, self.wavelet)
