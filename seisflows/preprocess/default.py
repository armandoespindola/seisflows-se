#!/usr/bin/env python3
"""
The SeisFlows Preprocessing module is in charge of interacting with seismic
data (observed and synthetic). It should contain functionality to read and write
seismic data, apply preprocessing such as filtering, quantify misfit,
and write adjoint sources that are expected by the solver.
"""
import os
import numpy as np
from glob import glob
from obspy import read as obspy_read
from obspy import Stream, Trace, UTCDateTime

from seisflows import logger
from seisflows.tools import signal, unix
from seisflows.tools.config import Dict, get_task_id

from seisflows.plugins.preprocess import misfit as misfit_functions
from seisflows.plugins.preprocess import adjoint as adjoint_sources
import sys


class Default:
    """
    Default Preprocess
    ------------------
    Data processing for seismic traces, with options for data misfit,
    filtering, normalization and muting.

    Parameters
    ----------
    :type obs_data_format: str
    :param obs_data_format: data format for reading observed traces into
        memory. Available formats: 'su', 'ascii', 'sac'
    :type unit_output: str
    :param unit_output: Data units. Must match the synthetic output of
        external solver. Available: ['DISP': displacement, 'VEL': velocity,
        'ACC': acceleration, 'PRE': pressure]
    :type misfit: str
    :param misfit: misfit function for waveform comparisons. For available
        see seisflows.plugins.preprocess.misfit
    :type backproject: str
    :param backproject: backprojection function for migration, or the
        objective function in FWI. For available see
        seisflows.plugins.preprocess.adjoint
    :type normalize: str
    :param normalize: Data normalization parameters used to normalize the
        amplitudes of waveforms. Choose from two sets:
        ENORML1: normalize per event by L1 of traces; OR
        ENORML2: normalize per event by L2 of traces;
        &
        TNORML1: normalize per trace by L1 of itself; OR
        TNORML2: normalize per trace by L2 of itself
    :type filter: str
    :param filter: Data filtering type, available options are:
        BANDPASS (req. MIN/MAX PERIOD/FREQ);
        LOWPASS (req. MAX_FREQ or MIN_PERIOD);
        HIGHPASS (req. MIN_FREQ or MAX_PERIOD)
    :type min_period: float
    :param min_period: Minimum filter period applied to time series.
        See also MIN_FREQ, MAX_FREQ, if User defines FREQ parameters, they
        will overwrite PERIOD parameters.
    :type max_period: float
    :param max_period: Maximum filter period applied to time series. See
        also MIN_FREQ, MAX_FREQ, if User defines FREQ parameters, they will
        overwrite PERIOD parameters.
    :type min_freq: float
    :param min_freq: Maximum filter frequency applied to time series,
        See also MIN_PERIOD, MAX_PERIOD, if User defines FREQ parameters,
        they will overwrite PERIOD parameters.
    :type max_freq: float
    :param max_freq: Maximum filter frequency applied to time series,
        See also MIN_PERIOD, MAX_PERIOD, if User defines FREQ parameters,
        they will overwrite PERIOD parameters.
    :type mute: list
    :param mute: Data mute parameters used to zero out early / late
        arrivals or offsets. Choose any number of:
        EARLY: mute early arrivals;
        LATE: mute late arrivals;
        SHORT: mute short source-receiver distances;
        LONG: mute long source-receiver distances

    Paths
    -----
    :type path_preprocess: str
    :param path_preprocess: scratch path for all preprocessing processes,
        including saving files
    ***
    """
    def __init__(self, syn_data_format="ascii", obs_data_format="ascii",
                 unit_output="DISP", misfit="waveform",
                 adjoint="waveform", normalize=None, filter=None,
                 min_period=None, max_period=None, min_freq=None, max_freq=None,
                 mute=None, early_slope=None, early_const=None, late_slope=None,
                 late_const=None, short_dist=None, long_dist=None,
                 workdir=os.getcwd(), path_preprocess=None, path_solver=None,
                 materials="acoustic",
                 **kwargs):
        """
        Preprocessing module parameters

        .. note::
            Paths and parameters listed here are shared with other modules and 
            so are not included in the class docstring.

        :type syn_data_format: str
        :param syn_data_format: data format for reading synthetic traces into
            memory. Shared with solver module. Available formats: 'su', 'ascii'
        :type workdir: str
        :param workdir: working directory in which to look for data and store
        results. Defaults to current working directory
        :type path_preprocess: str
        :param path_preprocess: scratch path for all preprocessing processes,
            including saving files
        """
        self.syn_data_format = syn_data_format.upper()
        self.obs_data_format = obs_data_format.upper()
        self.unit_output = unit_output.upper()
        self.misfit = misfit
        self.adjoint = adjoint
        self.normalize = normalize
        self.materials = materials

        self.filter = filter
        self.min_period = min_period
        self.max_period = max_period
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mute = mute or []
        self.normalize = normalize or []

        # Mute arrivals sub-parameters
        self.early_slope = early_slope
        self.early_const = early_const
        self.late_slope = late_slope
        self.late_const = late_const
        self.short_dist = short_dist
        self.long_dist = long_dist
        self.par = kwargs

        self.path = Dict(
            scratch=path_preprocess or os.path.join(workdir, "scratch",
                                                    "preprocess"),
            solver=path_solver or os.path.join(workdir, "scratch", "solver")
        )

        # The list <_obs_acceptable_data_formats> always includes
        # <_syn_acceptable_data_formats> in addition to more formats
        self._syn_acceptable_data_formats = ["SU", "ASCII"]
        self._obs_acceptable_data_formats = ["SU", "ASCII", "SAC"]

        self._acceptable_unit_output = ["DISP", "VEL", "ACC", "PRE"]

        # Misfits and adjoint sources are defined by the available functions
        # in each of these plugin files. Drop hidden variables from dir()
        self._acceptable_misfits = [_ for _ in dir(misfit_functions)
                                    if not _.startswith("_")]
        self._acceptable_adjsrcs = [_ for _ in dir(adjoint_sources)
                                    if not _.startswith("_")]

        # Internal attributes used to keep track of inversion workflows
        self._iteration = None
        self._step_count = None
        self._source_names = None

    def check(self):
        """ 
        Checks parameters and paths
        """
        if self.misfit:
            assert(self.misfit in self._acceptable_misfits), \
                f"preprocess.misfit must be in {self._acceptable_misfits}"
        if self.adjoint:
            assert(self.adjoint in self._acceptable_adjsrcs), \
                f"preprocess.misfit must be in {self._acceptable_adjsrcs}"

        # Data normalization option
        if self.normalize:
            acceptable_norms = {"TNORML1", "TNORML2", "ENORML1", "ENORML2"}
            chosen_norms = [_.upper() for _ in self.normalize]
            assert(set(chosen_norms).issubset(acceptable_norms))

        # Data muting options
        if self.mute:
            acceptable_mutes = {"EARLY", "LATE", "LONG", "SHORT"}
            chosen_mutes = [_.upper() for _ in self.mute]
            assert(set(chosen_mutes).issubset(acceptable_mutes))
            if "EARLY" in chosen_mutes:
                assert(self.early_slope is not None)
                assert(self.early_slope >= 0.)
                assert(self.early_const is not None)
            if "LATE" in chosen_mutes:
                assert(self.late_slope is not None)
                assert(self.late_slope >= 0.)
                assert(self.late_const is not None)
            if "SHORT" in chosen_mutes:
                assert(self.short_dist is not None)
                assert (self.short_dist > 0)
            if "LONG" in chosen_mutes:
                assert(self.long_dist is not None)
                assert (self.long_dist > 0)

        # Data filtering options that will be passed to ObsPy filters
        if self.filter:
            acceptable_filters = ["BANDPASS", "LOWPASS", "HIGHPASS"]
            assert self.filter.upper() in acceptable_filters, \
                f"self.filter must be in {acceptable_filters}"

            # Set the min/max frequencies and periods, frequency takes priority
            if self.min_freq is not None:
                self.max_period = 1 / self.min_freq
            elif self.max_period is not None:
                self.min_freq = 1 / self.max_period

            if self.max_freq is not None:
                self.min_period = 1 / self.max_freq
            elif self.min_period is not None:
                self.max_freq =  1 / self.min_period

            # Check that the correct filter bounds have been set
            if self.filter.upper() == "BANDPASS":
                assert(self.min_freq is not None and
                       self.max_freq is not None), \
                    ("BANDPASS filter PAR.MIN_PERIOD and PAR.MAX_PERIOD or " 
                     "PAR.MIN_FREQ and PAR.MAX_FREQ")
            elif self.filter.upper() == "LOWPASS":
                assert(self.max_freq is not None or
                       self.min_period is not None),\
                    "LOWPASS requires PAR.MAX_FREQ or PAR.MIN_PERIOD"
            elif self.filter.upper() == "HIGHPASS":
                assert(self.min_freq is not None or
                       self.max_period is not None),\
                    "HIGHPASS requires PAR.MIN_FREQ or PAR.MAX_PERIOD"

            # Check that filter bounds make sense, by this point, MIN and MAX
            # FREQ and PERIOD should be set, so we just check the FREQ
            assert(0 < self.min_freq < np.inf), "0 < PAR.MIN_FREQ < inf"
            assert(0 < self.max_freq < np.inf), "0 < PAR.MAX_FREQ < inf"
            assert(self.min_freq < self.max_freq), (
                "PAR.MIN_FREQ < PAR.MAX_FREQ"
            )

        assert(self.syn_data_format.upper() in self._syn_acceptable_data_formats), \
            f"synthetic data format must be in {self._syn_acceptable_data_formats}"

        assert(self.obs_data_format.upper() in self._obs_acceptable_data_formats), \
            f"observed data format must be in {self._obs_acceptable_data_formats}"

        assert(self.unit_output.upper() in self._acceptable_unit_output), \
            f"unit output must be in {self._acceptable_unit_output}"

    def setup(self):
        """
        Sets up data preprocessing machinery by dynamicalyl loading the
        misfit, adjoint source type, and specifying the expected file type
        for input and output seismic data.
        """
        unix.mkdir(self.path.scratch)

    def read(self, fid, data_format):
        """
        Waveform reading functionality. Imports waveforms as Obspy streams

        :type fid: str
        :param fid: path to file to read data from
        :type data_format: str
        :param data_format: format of the file to read data from
        :rtype: obspy.core.stream.Stream
        :return: ObsPy stream containing data stored in `fid`
        """
        st = None
        if data_format.upper() == "SU":
            st = obspy_read(fid, format="SU", byteorder="<")#,unpack_trace_headers=True)
            #ARMANDO : SU check in mu seconds (1e-3) units
            #for ist in st:
            #   ist.stats.delta *= 1000.0
            #logger.info(f"{st[0].stats.delta}")
        elif data_format.upper() == "SAC":
            st = obspy_read(fid, format="SAC")
        elif data_format.upper() == "ASCII":
            st = read_ascii(fid)
        return st

    def write(self, st, fid):
        """
        Waveform writing functionality. Writes waveforms back to format that
        SPECFEM recognizes

        :type st: obspy.core.stream.Stream
        :param st: stream to write
        :type fid: str
        :param fid: path to file to write stream to
        """
        if self.syn_data_format.upper() == "SU":
            for tr in st:
                # Work around for ObsPy data type conversion
                tr.data = tr.data.astype(np.float32)
            max_delta = 0.065535
            dummy_delta = max_delta

            if st[0].stats.delta > max_delta:
                for tr in st:
                    tr.stats.delta = dummy_delta

            # Write data to file
            st.write(fid, format="SU")

        elif self.syn_data_format.upper() == "ASCII":
            for tr in st:
                # Float provides time difference between starttime and default
                time_offset = float(tr.stats.starttime)
                data_out = np.vstack((tr.times() + time_offset, tr.data)).T
                np.savetxt(fid, data_out, ["%13.7f", "%17.7f"])

    def _calculate_misfit(self, **kwargs):
        """Wrapper for plugins.preprocess.misfit misfit/objective function"""
        if self.misfit is not None:
            return getattr(misfit_functions, self.misfit)(**kwargs)
        else:
            return None

    def _generate_adjsrc(self, **kwargs):
        """Wrapper for plugins.preprocess.adjoint source/backproject function"""
        if self.adjoint is not None:
            return getattr(adjoint_sources, self.adjoint)(**kwargs)
        else:
            return None

    def initialize_adjoint_traces(self, data_filenames, output,
                                  data_format=None):
        """
        SPECFEM requires that adjoint traces be present for every matching
        synthetic seismogram. If an adjoint source does not exist, it is
        simply set as zeros. This function creates all adjoint traces as
        zeros, to be filled out later

        Appends '.adj. to the solver filenames as expected by SPECFEM (if they
        don't already have that extension)

        TODO there are some sem2d and 3d specific tasks that are not carried
        TODO over here, were they required?

        :type data_filenames: list of str
        :param data_filenames: existing solver waveforms (synthetic) to read.
            These will be copied, zerod out, and saved to path `save`. Should
            come from solver.data_filenames
        :type output: str
        :param output: path to save the new adjoint traces to.
        """
        for fid in data_filenames:
            st = self.read(fid=fid, data_format=self.syn_data_format).copy()
            fid = os.path.basename(fid)  # drop any path before filename
            for tr in st:
                tr.data *= 0

            adj_fid = self._rename_as_adjoint_source(fid)

            # Write traces back to the adjoint trace directory
            self.write(st=st, fid=os.path.join(output, adj_fid))

    def _check_adjoint_traces(self, source_name, save_adjsrcs, synthetic):
        """Check that all adjoint traces required by SPECFEM exist"""
        source_name = source_name or self._source_names[get_task_id()]
        specfem_data_path = os.path.join(self.path.solver, source_name, "DATA")

        # since <STATIONS_ADJOINT> is generated only when using SPECFEM3D
        # by copying <STATIONS>, check adjoint stations in <STATIONS>
        adj_stations = np.loadtxt(os.path.join(specfem_data_path,
                                               "STATIONS"), dtype="str")

        if not isinstance(adj_stations[0], np.ndarray):
            adj_stations = [adj_stations]

        adj_template = "{net}.{sta}.{chan}.adj"

        if self.syn_data_format == 'SU':
            channels = [os.path.basename(syn).split('_')[0][1] for syn in synthetic]
            channels = list(set(channels))
        else:
            channels = [os.path.basename(syn).split('_')[2] for syn in synthetic]
            channels = list(set(channels))

        st = self.read(fid=synthetic[0], data_format=self.syn_data_format)
        for tr in st:
            tr.data *= 0.

        for adj_sta in adj_stations:
            sta = adj_sta[0]
            net = adj_sta[1]
            for chan in channels:
                adj_trace = adj_template.format(net=net, sta=sta, chan=chan)
                adj_trace = os.path.join(save_adjsrcs, adj_trace)
                if not os.path.isfile(adj_trace):
                    self.write(st=st, fid=adj_trace)

    def _rename_as_adjoint_source(self, fid):
        """
        Rename synthetic waveforms into filenames consistent with how SPECFEM
        expects adjoint sources to be named. Usually this just means adding
        a '.adj' to the end of the filename
        """
        if not fid.endswith(".adj"):
            if self.syn_data_format.upper() == "SU":
                fid = f"{fid}.adj"
            elif self.syn_data_format.upper() == "ASCII":
                # Differentiate between SPECFEM3D and 3D_GLOBE
                # SPECFEM3D: NN.SSSS.CCC.sem?
                # SPECFEM3D_GLOBE: NN.SSSS.CCC.sem.ascii
                ext = os.path.splitext(fid)[-1]  
                # SPECFEM3D
                if ".sem" in ext:
                    fid = fid.replace(ext, ".adj")
                # GLOBE (!!! Hardcoded to only work with ASCII format)
                elif ext == ".ascii":
                    root, ext1 = os.path.splitext(fid)  # .ascii
                    root, ext2 = os.path.splitext(root)  # .sem
                    fid = fid.replace(f"{ext2}{ext1}", ".adj")

        return fid

    def _setup_quantify_misfit(self, source_name):
        """
        Gather waveforms from the Solver scratch directory which will be used
        for generating adjoint sources
        """
        source_name = source_name or self._source_names[get_task_id()]

        obs_path = os.path.join(self.path.solver, source_name, "traces", "obs")
        syn_path = os.path.join(self.path.solver, source_name, "traces", "syn")

        observed = sorted(os.listdir(obs_path))
        synthetic = sorted(os.listdir(syn_path))



        assert(len(observed) != 0 and len(synthetic) != 0), \
            f"cannot quantify misfit, missing observed or synthetic traces"

        # verify observed traces format
        obs_ext = list(set([os.path.splitext(x)[-1] for x in observed]))
        print(obs_ext)

        
        if self.obs_data_format.upper() == "ASCII":
            obs_ext_ok = obs_ext[0].upper() == ".ASCII" or \
                         obs_ext[0].upper() == f".SEM{self.unit_output[0]}"
        else:
            obs_ext_ok = obs_ext[0].upper() == f".{self.obs_data_format}"

        assert(len(obs_ext) == 1 and obs_ext_ok), (
            f"observed traces have more than one format or their format "
            f"is not the one defined in parameters.yaml"
        )

        # verify synthetic traces format
        syn_ext = list(set([os.path.splitext(x)[-1] for x in synthetic]))
        

        if self.syn_data_format == "ASCII":
            syn_ext_ok = syn_ext[0].upper() == ".ASCII" or \
                         syn_ext[0].upper() == f".SEM{self.unit_output[0]}"
        else:
            syn_ext_ok = syn_ext[0].upper() == f".{self.syn_data_format}"

        assert(len(syn_ext) == 1 and syn_ext_ok), (
            f"synthetic traces have more than one format or their format "
            f"is not the one defined in parameters.yaml"
        )

        # remove data format
        observed = [os.path.splitext(x)[0] for x in observed]
        synthetic = [os.path.splitext(x)[0] for x in synthetic]

        
        
        
        # only return traces that have both observed and synthetic files
        matching_traces = sorted(list(set(synthetic).intersection(observed)))

        assert(len(matching_traces) != 0), (
            f"there are no traces with both observed and synthetic files for "
            f"source: {source_name}; verify that observations and synthetics "
            f"have the same name including channel code"
        )

        observed.clear()
        synthetic.clear()

        for file_name in matching_traces:
            observed.append(os.path.join(obs_path, f"{file_name}{obs_ext[0]}"))
            synthetic.append(os.path.join(syn_path, f"{file_name}{syn_ext[0]}"))

        assert(len(observed) == len(synthetic)), (
            f"number of observed traces does not match length of synthetic for "
            f"source: {source_name}"
        )

        return observed, synthetic

    def quantify_misfit(self, source_name=None, save_residuals=None,
                        export_residuals=None, save_adjsrcs=None, iteration=1,
                        step_count=0, **kwargs):
        """
        Prepares solver for gradient evaluation by writing residuals and
        adjoint traces. Meant to be called by solver.eval_func().

        Reads in observed and synthetic waveforms, applies optional
        preprocessing, assesses misfit, and writes out adjoint sources and
        STATIONS_ADJOINT file.

        TODO use concurrent futures to parallelize this

        :type source_name: str
        :param source_name: name of the event to quantify misfit for. If not
            given, will attempt to gather event id from the given task id which
            is assigned by system.run()
        :type save_residuals: str
        :param save_residuals: if not None, path to write misfit/residuls to
        :type save_adjsrcs: str
        :param save_adjsrcs: if not None, path to write adjoint sources to
        :type iteration: int
        :param iteration: current iteration of the workflow, information should
            be provided by `workflow` module if we are running an inversion.
            Defaults to 1 if not given (1st iteration)
        :type step_count: int
        :param step_count: current step count of the line search. Information
            should be provided by the `optimize` module if we are running an
            inversion. Defaults to 0 if not given (1st evaluation)
        """
        if self.par['se_double_difference']:
            logger.info('Double difference')
                       
        observed, synthetic = self._setup_quantify_misfit(source_name)
        if self.materials.upper() == "ANELASTIC":
            adjsrc_q = Stream()
        adjsrc = Stream()      

        #logger.info(f"{synthetic}")
        for obs_fid, syn_fid in zip(observed, synthetic):
            adjsrc = Stream()
            logger.info(f"{syn_fid}")
            logger.info(f"{obs_fid}")
            if not self.par['source_encoding']:
                obs = self.read(fid=obs_fid, data_format=self.obs_data_format)
            syn = self.read(fid=syn_fid, data_format=self.syn_data_format)

            #obs[0].plot()
            #syn[0].plot()

            
        
            # Process observations and synthetics identically
            if self.filter:
                if not self.par['source_encoding']:
                    obs = self._apply_filter(obs)
                syn = self._apply_filter(syn)
                logger.info("filter syn")
                #obs[0].plot()
                #syn[0].plot()
            if self.mute:
#                logger.info("Mute")
                if not self.par['source_encoding']:
                    obs = self._apply_mute(obs)
                #syn = self._apply_mute(syn)
                #obs[0].plot()
                #syn[0].plot()
            if self.normalize:
                if not self.par['source_encoding']:
                    obs = self._apply_normalize(obs)
                syn = self._apply_normalize(syn)
                #obs[0].plot()
                #syn[0].plot()


            if self.par['source_encoding']:
                syn_fid = os.path.basename(syn_fid)
                obs_fid = os.path.basename(obs_fid)

                logger.info(f"{syn_fid}")
                logger.info(f"{obs_fid}")

                self.prepare_obs_data_se(path_scratch = self.path.solver,
                                path_specfem_data = self.par['path_specfem_data'],
                                par = self.par,
                                fid = obs_fid)
                
                prepare_syn_data_se(path_scratch = self.path.solver,
                                path_specfem_data = self.par['path_specfem_data'],
                                syn_data = syn,
                                par = self.par,
                                fid = syn_fid)

                
                #syn_fid = synthetic[0]
                #syn = self.read(fid=syn_fid, data_format=self.syn_data_format)
                path = os.path.join(self.path.solver, source_name, "traces")
                fft_obs = np.load(path + "/{}_ft_obs.npy".format(obs_fid))
                t0_array = np.load(path + "/t0_array.npy")
                fft_syn = np.load(path + "/{}_ft_syn.npy".format(syn_fid))
                freq = np.load(self.par['path_specfem_data'] + "/es_freq.npy")
                rdi = np.load(self.par['path_specfem_data'] + "/es_rdi.npy")
                fft_stf = np.load(self.par['path_specfem_data'] + "/fft_stf.npy")
                freq_idx_glob = np.load(self.par['path_specfem_data'] + "/es_freq_idx_glob.npy")
                se_t = int(self.par['se_t'] / self.par['se_dwn'])
                se_td = int(self.par['se_td'] / self.par['se_dwn'])
                se_ntss = se_t - se_td
                se_dt = self.par['se_dt'] * self.par['se_dwn']
                nt_ss = se_ntss
                qf0  = self.par['qf0']
                #logger.info(f"EEEEEEEEEEEE{se_td}")



                # Compute Wp weight by frequency
                Wp = np.abs(fft_obs) * 0.0 + 1.0

                
                for istat in range(0,len(syn)):
                    obs_p = fft_obs[:,istat]
                    syn_p = fft_syn[:,istat]
                    ratio_p  = np.divide(syn_p, obs_p, out=np.zeros_like(syn_p), where=np.abs(obs_p)!=0)
                    phase_w = unwrap(np.angle(ratio_p))
                    Wp[:,istat] *= phase_w 
                    
                for ifreq in range(len(freq)):
                    obs_p = fft_obs[ifreq,:]
                    syn_p = fft_syn[ifreq,:]
                    ratio_p  = np.divide(syn_p, obs_p, out=np.zeros_like(syn_p), where=np.abs(obs_p)!=0)
                    phase_w = unwrap_d(np.angle(ratio_p),t0_array[ifreq,:])
                    # logger.info(f"T0: {t0_array[ifreq,:]}")
                    # logger.info(f"Phase: {phase_w}")
                    # phase_w[np.abs(phase_w) > 0] = 1.0
                    Wp[ifreq,:] *= phase_w
                    #logger.info(f"Wp: {Wp[ifreq,:]}")
                    # Andreas avoid uq in teh phase
                    # Wp[ifreq,:] = np.log(1 + np.abs(fft_obs[ifreq,:]))
                    # Wp[ifreq,:] /= np.max(Wp[ifreq,:])

                #Wp = np.log(1 + np.abs(fft_obs))
                #Wp /= np.max(Wp)


                for istat in range(0,len(syn)):
                    if self.mute or self.par['se_t0_mute']:
                        fft_syn [np.abs(fft_obs) == 0.0] = 0.0
                    obs_data = fft_obs[:,istat]
                    syn_data = fft_syn[:,istat]

                    #obs_data_max = np.max(np.abs(obs_data))
                    #obs_data[abs(obs_data) < obs_data_max * 1e-2] = 0.0
                    #syn_data[abs(obs_data) < obs_data_max * 1e-2] = 0.0

                    # import matplotlib
                    # import matplotlib.pyplot as plt
                    # matplotlib.use('Agg')

                    # if istat < len(freq):
                    #     plt.figure()
                    #     plt.plot(np.angle(fft_obs[istat,:]),'ko-')
                    #     plt.plot(np.angle(fft_syn[istat,:]),'r*-')
                    #     plt.savefig(f"phase_test_{istat}.png")
                    # if istat < len(freq):
                    #     plt.figure()
                    #     plt.plot(np.real(fft_obs[istat,:]),'ko-')
                    #     plt.plot(np.real(fft_syn[istat,:]),'b*-')
                    #     plt.savefig(f"real_test_{istat}.png")
                    # plt.plot(np.unwrap(np.angle(fft_obs[90,:]) * Wp[90,:]),'ko-')
                    # plt.plot(np.unwrap(np.angle(fft_syn[90,:]) * Wp[90,:]),'b*-')
                    # plt.plot(np.angle(fft_syn[0,:]/fft_obs[90,:]) * Wp[90,:],'go-')
                    # plt.figure()
                    # plt.plot(np.real(obs_data - syn_data),'ko-')
                    # plt.plot(np.imag(obs_data - syn_data),'b*-')

             #       plt.show()
                    # Simple check to make sure zip retains ordering
                    #tr_obs.plot()
                    #tr_syn.plot()
                    # Calculate the misfit value and write to file
                    #logger.info(f"{se_ntss},{se_dt},{nt_ss},{len(syn[0].data)}")
                    if os.path.basename(syn_fid)[1].upper() in list(self.par['components']):
                        #logger.info(f"computing misfit and adjoint {syn_fid} - {obs_fid}")
                        if save_residuals and self._calculate_misfit:
                            residual,diff = self._calculate_misfit(
                                obs=obs_data, syn=syn_data)

                            if self.par['se_double_difference']:
                                #logger.info('Double difference')
                                residual = 0.0
                                diff_sum = np.zeros(len(diff))
                                for jstat in range(istat - 5,istat + 5):
                                    if istat != jstat and jstat > 0 and jstat < len(syn):
                                        _,diff2 = self._calculate_misfit(obs=fft_obs[:,jstat],
                                                                           syn=fft_syn[:,jstat])
                                        diff_sum = np.nansum([diff_sum,diff2],axis=0)

                                diff = -1.0 * (diff - diff_sum)
                                residual = 0.5 * np.sqrt(np.sum(np.multiply(diff,diff)))
                            #logger.info(f"{residual}")
                
                            with open(save_residuals, "a") as f:
                                f.write(f"{residual:.2E}\n")

                        # Generate an adjoint source trace, write to file
                        if save_adjsrcs and self._generate_adjsrc:
                            adjsrc_st = syn[istat].copy()
                            adjsrc_st.data[:] = 0.0
                            adjsrc_st.data = self._generate_adjsrc(
                                obs=obs_data, syn=syn_data,
                                se_t = se_t, se_td = se_td, se_tse = se_ntss * se_dt,
                                se_dt = se_dt, nt_se = nt_ss,freq = freq, freq_idx = freq_idx_glob,
                                rdi = rdi, fft_stf = fft_stf,gamma = self.par['se_gamma'],t0_array=t0_array[:,istat],
                                Wp = Wp[:,istat],dd_r=diff,dd_diff=self.par['se_double_difference'])

                            if self.materials.upper() == "ANELASTIC":
                                adjsrc_st_q = syn[istat].copy()
                                adjsrc_st_q.data[:] = 0.0
                                adj_data =  adjsrc_st.data.copy()
                                adjsrc_st_q.data = elastic_to_anelastic_adj(adj_data, se_dt, qf0,freq_idx_glob,
                                                                            self.par['se_gamma'])

                            #adjsrc_st.resample(sampling_rate= 0.50 / adjsrc_st.stats.delta)
                            #logger.info(f"{adjsrc_st.stats}")
                            #adjsrc_st.plot()
                    else:
                        adjsrc_st = syn[istat].copy()
                        adjsrc_st.data[:] = 0.0

                    if self.materials.upper() == "ANELASTIC":
                        adjsrc_q.append(adjsrc_st_q)
                    adjsrc.append(adjsrc_st)

                adjsrc.resample(sampling_rate = 1.0 / self.par['se_dt'] )
                adjsrc.taper(0.05,side='right')
                fid = os.path.basename(syn_fid)
                fid = self._rename_as_adjoint_source(fid) + "_e"
                logger.info(f"writing adjsource {os.path.join(save_adjsrcs, fid)}")
                self.write(st=adjsrc, fid=os.path.join(save_adjsrcs, fid))


                if self.materials.upper() == "ANELASTIC":
                    adjsrc_q.resample(sampling_rate = 1.0 / self.par['se_dt'] )
                    adjsrc_q.taper(0.05,side='right')
                    fid = os.path.basename(syn_fid)
                    fid = self._rename_as_adjoint_source(fid) + "_q"
                    logger.info(f"writing adjsource {os.path.join(save_adjsrcs, fid)}")
                    self.write(st=adjsrc_q, fid=os.path.join(save_adjsrcs, fid))

                if save_adjsrcs and self._generate_adjsrc:
                    self._check_adjoint_traces(source_name, save_adjsrcs, synthetic)

                # Exporting residuals to disk (output/) for more permanent storage
                if export_residuals:
                    if not os.path.exists(export_residuals):
                        unix.mkdir(export_residuals)
                        unix.cp(src=save_residuals, dst=export_residuals)
                               
            
            continue
        
        
            # Write the residuals/misfit and adjoint sources for each component
            #logger.info("gsgsgsgsggsg")
            for tr_obs, tr_syn in zip(obs, syn):
                
                obs_data = tr_obs.data
                syn_data = tr_syn.data
                
                    
                # Simple check to make sure zip retains ordering
                #tr_obs.plot()
                #tr_syn.plot()
                assert(tr_obs.stats.component == tr_syn.stats.component)
                # Calculate the misfit value and write to file
                if save_residuals and self._calculate_misfit:
                    residual = self._calculate_misfit(
                        obs=obs_data, syn=syn_data,
                        nt=tr_syn.stats.npts, dt=tr_syn.stats.delta
                    )
                    with open(save_residuals, "a") as f:
                        f.write(f"{residual:.2E}\n")

                # Generate an adjoint source trace, write to file
                if save_adjsrcs and self._generate_adjsrc:
                    adjsrc_st = tr_syn.copy()
                    adjsrc_st.data[:] = 0.0
                    adjsrc_st.data = self._generate_adjsrc(
                        obs=obs_data, syn=syn_data,
                        nt=tr_syn.stats.npts, dt=tr_syn.stats.delta)
                    #adjsrc_st.plot()
                    adjsrc.append(adjsrc_st)
                    fid = os.path.basename(syn_fid)
                    fid = self._rename_as_adjoint_source(fid)
                    self.write(st=adjsrc, fid=os.path.join(save_adjsrcs, fid))

        if save_adjsrcs and self._generate_adjsrc:
            self._check_adjoint_traces(source_name, save_adjsrcs, synthetic)

        # Exporting residuals to disk (output/) for more permanent storage
        if export_residuals:
            if not os.path.exists(export_residuals):
                unix.mkdir(export_residuals)
            unix.cp(src=save_residuals, dst=export_residuals)

    def finalize(self):
        """Teardown procedures for the default preprocessing class"""
        pass

    @staticmethod
    def sum_residuals(residuals):
        """
        Returns the summed square of residuals for each event. Following
        Tape et al. 2007

        :type residuals: np.array
        :param residuals: list of residuals from each NTASK event
        :rtype: float
        :return: sum of squares of residuals
        """
        return np.sum(residuals ** 2.)

    def _apply_filter(self, st):
        """
        Apply a filter to waveform data using ObsPy

        :type st: obspy.core.stream.Stream
        :param st: stream to be filtered
        :rtype: obspy.core.stream.Stream
        :return: filtered traces
        """
        # Pre-processing before filtering
        st.detrend("demean")
        st.detrend("linear")
        st.taper(0.05, type="hann")

        

        if self.filter.upper() == "BANDPASS":
            st.filter("bandpass", zerophase=True, freqmin=self.min_freq,
                      freqmax=self.max_freq)
        elif self.filter.upper() == "LOWPASS":
            #logger.info(f"{self.max_freq} EEEE")
            st.filter("lowpass", zerophase=True, freq=self.max_freq)
        elif self.filter.upper() == "HIGHPASS":
            st.filter("highpass", zerophase=True, freq=self.min_freq)

#        st[0].plot()
#        logger.info(f"{st[0].stats}")

        return st

    def _apply_mute(self, st):
        """
        Apply mute on data based on early or late arrivals, and short or long
        source receiver distances

        .. note::
            The underlying mute functions have been refactored but not tested
            as I was not aware of the intended functionality. Not gauranteed
            to work, use at your own risk.

        :type st: obspy.core.stream.Stream
        :param st: stream to mute
        :rtype: obspy.core.stream.Stream
        :return: muted stream object
        """
        mute_choices = [_.upper() for _ in self.mute]
        if "EARLY" in mute_choices:
            st = signal.mute_arrivals(st, slope=self.early_slope,
                                      const=self.early_const, choice="EARLY")
        if "LATE" in mute_choices:
            st = signal.mute_arrivals(st, slope=self.late_slope,
                                      const=self.late_const, choice="LATE")
        if "SHORT" in mute_choices:
            st = signal.mute_offsets(st, dist=self.short_dist, choice="SHORT")
        if "LONG" in mute_choices:
            st = signal.mute_offsets(st, dist=self.long_dist, choice="LONG")

        return st

    def _apply_normalize(self, st):
        """
        Normalize the amplitudes of waveforms based on user choice

        .. note::
            The normalization function has been refactored but not tested
            as I was not aware of the intended functionality. Not gauranteed
            to work, use at your own risk.

        :type st: obspy.core.stream.Stream
        :param st: All of the data streams to be normalized
        :rtype: obspy.core.stream.Stream
        :return: stream with normalized traces
        """
        st_out = st.copy()
        norm_choices = [_.upper() for _ in self.normalize]

        # Normalize an event by the L1 norm of all traces
        if 'ENORML1' in norm_choices:
            w = 0.
            for tr in st_out:
                w += np.linalg.norm(tr.data, ord=1)
            for tr in st_out:
                tr.data /= w
        # Normalize an event by the L2 norm of all traces
        elif "ENORML2" in norm_choices:
            w = 0.
            for tr in st_out:
                w += np.linalg.norm(tr.data, ord=2)
            for tr in st_out:
                tr.data /= w
        # Normalize each trace by its L1 norm
        if "TNORML1" in norm_choices:
            for tr in st_out:
                w = np.linalg.norm(tr.data, ord=1)
                if w > 0:
                    tr.data /= w
        elif "TNORML2" in norm_choices:
            # normalize each trace by its L2 norm
            for tr in st_out:
                w = np.linalg.norm(tr.data, ord=2)
                if w > 0:
                    tr.data /= w

        return st_out


    def prepare_obs_data_se(self,path_scratch,path_specfem_data,par,fid):
        from scipy.fft import fft,fftfreq
        import numpy as np
        import obspy
    
        freq = np.load(path_specfem_data + "/es_freq.npy")
        rdi = np.load(path_specfem_data + "/es_rdi.npy")
        freq_idx_glob = np.load(path_specfem_data + "/es_freq_idx_glob.npy")
        fft_stf = np.load(path_specfem_data + "/fft_stf.npy")
        se_ntss = int((par['se_t'] - par['se_td']) / par['se_dwn'])
        logger.info("Preparing obs data for source encoding")
        if (par['se_t0']):
            logger.info("Using gamma * t0 damping")
        #logger.info(f"{fid}")
        fftobs_full = [] #np.zeros((len(freq),nstation),dtype=complex)
        t0_array = []
        for ifreq in range(0,len(freq)):
            source_name = "{:03}".format(rdi[ifreq] + 1)
            path = os.path.join(path_scratch,source_name,"traces","obs")
            #logger.info(f"{os.path.join(path,fid)}")
            obs_data = self.read(fid=os.path.join(path,fid),
                                 data_format=self.obs_data_format)
            #logger.info("OKK")
            if self.filter:    
                obs_data = self._apply_filter(obs_data)
                logger.info("filt obs")
            if self.mute:
#                logger.info("Mute-Obs")
                obs_data = self._apply_mute(obs_data)
            if self.normalize:
                obs_data = self._apply_normalize(obs_data)

            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # obs_data_raw = obs_data.copy()
            for ir,tr_obs in enumerate(obs_data):
                data_obs = tr_obs.data.copy()
                
                dt = tr_obs.stats.delta
                ntss = se_ntss

                if len(data_obs) < ntss:
                    ntaper = int(len(data_obs) * 0.025)
                    taper = np.hanning(ntaper * 2)
                    data_obs[:ntaper] *= taper[:ntaper]
                    data_obs[-ntaper:] *= taper[-ntaper:]
                    data_obs = np.pad(data_obs,(0,ntss - len(data_obs)),constant_values=(0, 0))
                    ksample = 1
                else:
                    ksample = np.int(np.ceil(len(data_obs) / ntss))
                    ntaper = int(len(data_obs) * 0.025)
                    taper = np.hanning(ntaper * 2)
                    data_obs[:ntaper] *= taper[:ntaper]
                    data_obs[-ntaper:] *= taper[-ntaper:]
                    data_obs = np.pad(data_obs,(0,ksample * ntss - len(data_obs)),constant_values=(0, 0))


                sx = tr_obs.stats.su.trace_header.source_coordinate_x
                sy = tr_obs.stats.su.trace_header.source_coordinate_y
                rx = tr_obs.stats.su.trace_header.group_coordinate_x
                ry = tr_obs.stats.su.trace_header.group_coordinate_y
                distance = np.sqrt((sx - rx)**2 + (sy - ry)**2) * (sx - rx) / np.abs(sx - rx)
                # tr_obs.stats.distance = distance
                # obs_data_raw[ir].stats.distance = distance
                if (par['se_t0']):
                    #logger.info("Using gamma * t0 damping")
                    t0 = pick_t0(data_obs,tr_obs.stats.delta,1.0 / par['se_max_freq'])
                else:
                    t0 = 0.0
                #logger.info(f"Muting {t0}")
                #plt.plot(distance/1000,t0,"ro")
                #data_obs_old = data_obs.copy()
                if (par['se_t0_mute']) and (par['se_t0']):
                    if t0 < par['se_t0_min'] or t0 > par['se_t0_max']:
                        #logger.info(f"Muting {t0}")
                        t0 = 0.0
                        data_obs = data_obs * 0.0
                    
                #logger.info(f"Muting {t0}")
                data_obs *= np.exp(-1.0 * par['se_gamma'] * (np.arange(len(data_obs)) * dt - t0))
                data_obs *= np.exp(-1.0 * par['se_gamma'] * 1.20 / freq[ifreq])

                t0_array.append(t0)
                # tr_obs.data = data_obs
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.plot(data_obs_old,'k')
                # plt.plot(data_obs,'r')
                # plt.plot(data_obs_old * np.exp(-1.0 * par['se_gamma'] * (np.arange(len(data_obs)) * par['se_dt'])) ,'b')
                # plt.show()

                
                fft_obs  = fft(data_obs)[::ksample] #* np.exp(t0 * par['se_gamma'])
                freq_obs = np.fft.fftfreq(len(fft_obs),dt)
                #factor = np.exp(1j * freq_obs * 2.0 * np.pi * td)
                factor = np.exp(1j * freq_obs * 2.0 * np.pi * dt * par['se_td'] / par['se_dwn'])
                factor *= np.exp(-1j * freq_obs * 2.0 * np.pi * 1.20 / freq[ifreq])
                fft_obs *= factor  * -1.0j
#                fft_obs[freq_idx_glob] /= fft_stf
                fftobs_full.append(fft_obs[freq_idx_glob[ifreq]]/ fft_stf[ifreq])
            #obs_data_raw.plot(type='section',fig=fig,color='red')
            #obs_data.plot(type='section',fig=fig)
            #plt.ylim((0,10))
            #plt.show()
                #t0_array.append(pick_t0(data_obs,tr_obs.stats.delta,par['se_min_freq']))
#            logger.info(f"{source_name} -> freq:  {freq_obs[freq_idx_glob[ifreq]]} , {freq[ifreq]} ")
        nfreq = len(freq); nstation = int(len(fftobs_full) / len(freq))
        fftobs_full = np.reshape(np.array(fftobs_full),(nfreq,nstation))
        t0_array = np.reshape(np.array(t0_array),(nfreq,nstation))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for i in range(0,len(freq)):
        #     plt.plot(t0_array[i,:])
        # plt.show()
        # #logger.info(f"{fftobs_full.shape}")
        path = os.path.join(path_scratch,"001","traces")
        np.save(os.path.join(path, fid + "_ft_obs"),fftobs_full)
        np.save(os.path.join(path, "t0_array"),t0_array)
        



def read_ascii(fid, origintime=None):
    """
    Read waveforms in two-column ASCII format. This is copied directly from
    pyatoa.utils.read.read_sem()
    """
    try:
        times = np.loadtxt(fname=fid, usecols=0)
        data = np.loadtxt(fname=fid, usecols=1)

    # At some point in 2018, the Specfem developers changed how the ascii files
    # were formatted from two columns to comma separated values, and repeat
    # values represented as 2*value_float where value_float represents the data
    # value as a float
    except ValueError:
        times, data = [], []
        with open(fid, 'r') as f:
            lines = f.readlines()
        for line in lines:
            try:
                time_, data_ = line.strip().split(',')
            except ValueError:
                if "*" in line:
                    time_ = data_ = line.split('*')[-1]
                else:
                    raise ValueError
            times.append(float(time_))
            data.append(float(data_))

        times = np.array(times)
        data = np.array(data)

    if origintime is None:
        origintime = UTCDateTime("1970-01-01T00:00:00")

    # We assume that dt is constant after 'precision' decimal points
    delta = round(times[1] - times[0], 4)

    # Honor that Specfem doesn't start exactly on 0
    origintime += times[0]

    # Write out the header information. Deal with the fact that SPECFEM2D/3D and
    # 3D_GLOBE have slightly different formats for their filenames
    net, sta, cha, *fmt = os.path.basename(fid).split('.')
    stats = {"network": net, "station": sta, "location": "",
             "channel": cha, "starttime": origintime, "npts": len(data),
             "delta": delta, "mseed": {"dataquality": 'D'},
             "time_offset": times[0], "format": fmt[0]
             }
    st = Stream([Trace(data=data, header=stats)])

    return st







def sta_lta(data, dt, min_period):
    from scipy.signal import lfilter
    """
    STA/LTA as used in FLEXWIN.

    :param data: The data array.
    :param dt: The sample interval of the data.
    :param min_period: The minimum period of the data.
    """
    Cs = 10 ** (-dt / min_period)
    Cl = 10 ** (-dt / (12 * min_period))
    TOL = 1e-9

    noise = data.max() / 1E5

    # 1000 samples should be more then enough to "warm up" the STA/LTA.
    extended_syn = np.zeros(len(data) + 1000, dtype=np.float64)
    # copy the original synthetic into the extended array, right justified
    # and add the noise level.
    extended_syn += noise
    extended_syn[-len(data):] += data

    # This piece of codes "abuses" SciPy a bit by "constructing" an IIR
    # filter that does the same as the decaying sum and thus avoids the need to
    # write the loop in Python. The result is a speedup of up to 2 orders of
    # magnitude in common cases without needing to write the loop in C which
    # would have a big impact in the ease of installation of this package.
    # Other than that its quite a cool little trick.
    a = [1.0, -Cs]
    b = [1.0]
    sta = lfilter(b, a, extended_syn)

    a = [1.0, -Cl]
    b = [1.0]
    lta = lfilter(b, a, extended_syn)

    # STA is now STA_LTA
    sta /= lta

    # Apply threshold to avoid division by very small values.
    sta[lta < TOL] = noise
    return sta[-len(data):]

def pick_t0(data,dt,min_period,thr_1=0.6,thr_2=0.3):
    from obspy.signal.trigger import trigger_onset
    import matplotlib.pyplot as plt
    cft = np.abs(sta_lta(data, dt, min_period))
    on_off = trigger_onset(cft,thr_1,thr_2)
    # fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    # axs[0].plot(cft)
    # axs[1].plot(data, 'k')
    # axs[1].vlines(on_off[0, 0],-np.max(data),np.max(data), color='r', linewidth=2)
    # plt.show()
    #logger.info(f"{on_off}")
    if len(on_off) > 0:
        t0 = on_off[0,0] * dt
        #t0 = 0.0
    else:
        #fig, axs = plt.subplots(1, 2, figsize=(9, 3))
        #axs[0].plot(cft)
        #axs[1].plot(data, 'k')
        #axs[1].vlines(on_off[0, 0],-np.max(data),np.max(data), color='r', linewidth=2)
        #plt.show()
        #logger.info(f"{on_off}")
        t0 = 0.0
    return t0
                    
                    


def prepare_syn_data_se(path_scratch,path_specfem_data,syn_data,par,fid):
    from scipy.fft import fft,fftfreq
    import numpy as np
    import obspy
    freq = np.load(path_specfem_data + "/es_freq.npy")
    rdi = np.load(path_specfem_data + "/es_rdi.npy")
    freq_idx_glob = np.load(path_specfem_data + "/es_freq_idx_glob.npy")
    se_ntss = int((par['se_t'] - par['se_td']) / par['se_dwn'])
    fft_stf = np.load(path_specfem_data + "/fft_stf.npy")
    nstation = len(syn_data)
    logger.info("Preparing syn data for source encoding")
    fftsyn_full = np.zeros((len(freq),nstation),dtype=complex)

    path = os.path.join(path_scratch,"001","traces")
    t0_array = np.load(path + "/t0_array.npy")
    
    path = os.path.join(path_scratch,"solver","001","traces","syn")    

    #syn_data.resample(sampling_rate= 2.0 / syn_data[0].stats.delta)
    for ir,tr_syn in enumerate(syn_data):
        dt = tr_syn.stats.delta
        ntss = se_ntss
        #logger.info(f"{ntse}")
        #logger.info(f"{dt}")
        data_syn = tr_syn.data[-ntss:]

        # gamma
        t0 = t0_array[:,ir]
        #for it0 in t0:
        data_syn *= np.exp(-1 * par['se_gamma'] * (np.arange(len(data_syn)) * dt 
                                                   + dt * par['se_td'] / par['se_dwn']))
        
        
        fft_syn  = fft(data_syn)
        freq_syn = fftfreq(ntss,dt)
        # Compensating by TD transient duration
        #fft_syn *= np.exp(-1j * freq_syn * 2 * np.pi * dt * par['se_td'] / par['se_dwn'])

        # Compensate sin to cosine Ricker wavelet
        #fft_syn *=  1j

        # Compensate for dt and integraton 2 / dtao
        fft_syn *= 2.0 / ntss 

            # # Compensate for source
            #logger.info(f"{stf.shape}")                
            #fft_stf = fft(stf[:ntse,1])

            #fft_syn *= fft_stf * 1.0e-10

        fftsyn_full[:,ir] = fft_syn[freq_idx_glob] * np.exp(par['se_gamma'] * t0)
#        fftsyn_full[:,ir] = fft_syn[freq_idx_glob] * fft_stf * np.exp(par['se_gamma'] * t0)
        
        #fftsyn_full[:,ir] = fft_syn[freq_idx_glob] * np.exp(par['se_gamma'] * t0)
        

    path = os.path.join(path_scratch,"001","traces")
    #logger.info(f"{path}")                
    np.save(os.path.join(path, fid + "_ft_syn"),fftsyn_full)




def unwrap_d(data,t0):
    ''' get unwrapped  angle the data according to the traveltime.
        We will apply np.unwrap according to the traveltime. 
        INPUT
            data        - 1d data array with size of number of receivers
            t0          - 1d traveltime array with size of number of receivers
    '''
    phase_unwrap = np.zeros(data.shape)
    # Firstly, we will find the index when t0>0 
    idxgreat0 = (t0>0)
    t0fil = t0[idxgreat0]
    idxmin = np.argmin(abs(t0fil))
    print("t0_idx: ",idxmin)
    #logger.info(f"idx: {idxmin}")
    data_sel = data[idxgreat0]
    phase_ww = np.ones(len(data_sel))
    phase_temp = np.zeros(data_sel.shape)

    
    for i in range(len(data_sel)):
        if i < idxmin:
            if abs(data_sel[i]) > 1.6 or np.isnan(data_sel[i]):
                phase_ww[:i + 1:] = np.nan
                
        if i > idxmin:
            if abs(data_sel[i]) > 1.6 or np.isnan(data_sel[i]):
                phase_ww[i::] = np.nan


    if abs(data_sel[idxmin]) > 1.6 or np.isnan(data_sel[idxmin]):
        phase_ww[:] = np.nan
                
    phase_unwrap[idxgreat0] = phase_ww
    
    return phase_unwrap


def unwrap(data_phase):
    phase_w = np.ones(len(data_phase))
    for i in range(len(data_phase)):
        if abs(data_phase[i]) > np.pi / 2.0:
            phase_w[i] = np.nan
            phase_w[i:] = np.nan
    return phase_w



def elastic_to_anelastic_adj(adj_data, dt, f0,freq_idx,gamma):
    # from scipy.fftpack import hilbert as hilbert_transform
    twopi = 2*np.pi
    w0 = twopi*f0
    f = adj_data  # forward in time
    f *= np.exp(1.0 * gamma * (np.arange(len(f)) * dt))
    # f = adj.adjoint_source
    F = np.fft.fft(f)
    freqs = np.fft.fftfreq(len(f), d=dt)
    w = twopi*freqs
    # fig, ax = plt.subplots()
    w[0] = w0
    # phy = np.conj((2/np.pi)*np.l# o
    w_zero = np.zeros(len(f))
    w_zero[freq_idx] = 1.0
    w_zero[-freq_idx] = 1.0
    phy = (2/np.pi)*np.log(np.abs(w)/w0)
    # g(np.abs(w)/w0))
    # phy = np.log(np.abs(w)/w0)
    phy[0] = phy[1]
    phy *= w_zero
    atten_adj_source_1 = np.real(np.fft.ifft(F*phy))

    amp = -1j*np.sign(w) * w_zero
    atten_adj_source_2 = np.real(np.fft.ifft(F*amp))
    atten_adj_source = atten_adj_source_1 + atten_adj_source_2

    atten_adj_source *= np.exp(-1.0 * gamma * (np.arange(len(f)) * dt))

    return atten_adj_source
