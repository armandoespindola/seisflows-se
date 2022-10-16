#!/usr/bin/env python3
"""
This class provides utilities for the Seisflows solver interactions with
Specfem3D Cartesian.
"""
import os
from glob import glob
from seisflows import logger
from seisflows.tools import unix
from seisflows.tools.specfem import setpar, getpar
from seisflows.solver.specfem import Specfem


class Specfem3D(Specfem):
    """
    Solver SPECFEM3D
    ----------------
    SPECFEM3D-specific alterations to the base SPECFEM module

    Parameters
    ----------
    :type source_prefix: str
    :param source_prefix: Prefix of source files in path SPECFEM_DATA. Must be
        in ['CMTSOLUTION', 'FORCESOLUTION']. Defaults to 'CMTSOLUTION'
    :type export_vtk: bool
    :param export_vtk: anytime a model, kernel or gradient is considered,
        generate a VTK file and store it in the scratch/ directory for the User
        to visualize at their leisure.
    :type prune_scratch: bool
    :param prune_scratch: prune/remove database files as soon as they are used,
        to keep overall filesystem burden down
        - removes *.vt? files after they're generated by a forward simulation
        - removes proc*_absorb_field.bin and proc*_save_forward_array.bin
            files after adjoint simulations

    Paths
    -----
    ***
    """
    __doc__ = Specfem.__doc__ + __doc__

    def __init__(self, source_prefix="CMTSOLUTION", export_vtk=True,
                 prune_scratch=True, **kwargs):
        """Instantiate a Specfem3D_Cartesian solver interface"""

        super().__init__(source_prefix=source_prefix, **kwargs)

        self.prune_scratch = prune_scratch
        self.export_vtk = export_vtk

        # Define parameters based on material type
        if self.materials.upper() == "ACOUSTIC":
            self._parameters += ["vp"]
        elif self.materials.upper() == "ELASTIC":
            self._parameters += ["vp", "vs"]

        # Overwriting the base class parameters
        self._acceptable_source_prefixes = ["CMTSOLUTION", "FORCESOLUTION"]
        self._required_binaries = ["xspecfem3D", "xmeshfem3D",
                                   "xgenerate_databases", "xcombine_sem",
                                   "xsmooth_sem", "xcombine_vol_data_vtk"]

        self._model_databases = None
        self.path._vtk_files = os.path.join(self.path.scratch, "vtk_files")

    def setup(self):
        """
        Generate .vtk files for the initial and target (if applicable) models,
        which the User can use for external visualization
        """
        super().setup()

        # Work-in-progress
        # self.combine_vol_data_vtk()

    def data_wildcard(self, comp="?"):
        """
        Returns a wildcard identifier for synthetic data

        TODO where does SU put its component?

        :rtype: str
        :return: wildcard identifier for channels
        """
        if self.data_format.upper() == "SU":
            return f"*_d?_SU"
        elif self.data_format.upper() == "ASCII":
            return f"*.?X{comp}.sem?"

    @property
    def model_databases(self):
        """
        The location of databases for model outputs, usually
        OUTPUT_FILES/DATABASES_MPI. This can be determined by 'LOCAL_PATH'
        in your Par_file
        """
        if self._model_databases is None:
            self._model_databases = getpar(
                key="LOCAL_PATH", file=os.path.join(self.path.specfem_data,
                                                    "Par_file"))[1]
        return self._model_databases

    @property
    def kernel_databases(self):
        """
        The location of databases for kernel outputs, usually the same as
        'model_databases'
        """
        return self.model_databases

    def forward_simulation(self, executables=None, save_traces=False,
                           export_traces=False, **kwargs):
        """
        Calls SPECFEM3D forward solver, exports solver outputs to traces dir

        :type executables: list or None
        :param executables: list of SPECFEM executables to run, in order, to
            complete a forward simulation. This can be left None in most cases,
            which will select default values based on the specific solver
            being called (2D/3D/3D_GLOBE). It is made an optional parameter
            to keep the function more general for inheritance purposes.
        :type save_traces: str
        :param save_traces: move files from their native SPECFEM output location
            to another directory. This is used to move output waveforms to
            'traces/obs' or 'traces/syn' so that SeisFlows knows where to look
            for them, and so that SPECFEM doesn't overwrite existing files
            during subsequent forward simulations
        :type export_traces: str
        :param export_traces: export traces from the scratch directory to a more
            permanent storage location. i.e., copy files from their original
            location
        """
        unix.cd(self.cwd)

        if executables is None:
            executables = ["bin/xgenerate_databases", "bin/xspecfem3D"]

            # Database files only need to be made once, usually at the first
            # evaluation. Once made, we don't have to run xmeshfem3D anymore.
            if not glob(os.path.join(self.model_databases, "proc*_Database")):
                executables = ["bin/xmeshfem3D"] + executables

        # SPECFEM3D has to deal with attenuation
        if self.attenuation:
            setpar(key="ATTENUATION", val=".true.", file="DATA/Par_file")
        else:
            setpar(key="ATTENUATION", val=".false`.", file="DATA/Par_file")

        super().forward_simulation(executables=executables,
                                   save_traces=save_traces,
                                   export_traces=export_traces
                                   )

        if self.prune_scratch:
            logger.debug("removing '*.vt?' files from database directory")
            unix.rm(glob(os.path.join(self.model_databases, "proc*_*.vt?")))

    def adjoint_simulation(self, executables=None, save_kernels=False,
                           export_kernels=False):
        """
        Calls SPECFEM3D adjoint solver, creates the `SEM` folder with adjoint
        traces which is required by the adjoint solver

        :type executables: list or None
        :param executables: list of SPECFEM executables to run, in order, to
            complete an adjoint simulation. This can be left None in most cases,
            which will select default values based on the specific solver
            being called (2D/3D/3D_GLOBE). It is made an optional parameter
            to keep the function more general for inheritance purposes.
        :type save_kernels: str
        :param save_kernels: move the kernels from their native SPECFEM output
            location to another path. This is used to move kernels to another
            SeisFlows scratch directory so that they are discoverable by
            other modules. The typical location they are moved to is
            path_eval_grad
        :type export_kernels: str
        :param export_kernels: export/copy/save kernels from the scratch
            directory to a more permanent storage location. i.e., copy files
            from their original location. Note that kernel file sizes are LARGE,
            so exporting kernels can lead to massive storage requirements.
        """
        if executables is None:
            executables = ["bin/xspecfem3D"]

        # Make sure attenuation is OFF, if ON you'll get a floating point error
        unix.cd(self.cwd)
        setpar(key="ATTENUATION", val=".false.", file="DATA/Par_file")

        # Make sure we have a STATIONS_ADJOINT file. Simply copy STATIONS file
        # !!! Do we need to tailor this to output of preprocess module? !!!
        dst = os.path.join(self.cwd, "DATA", "STATIONS_ADJOINT")
        if not os.path.exists(dst):
            src = os.path.join(self.cwd, "DATA", "STATIONS")
            unix.cp(src, dst)

        super().adjoint_simulation(executables=executables,
                                   save_kernels=save_kernels,
                                   export_kernels=export_kernels)

        if self.prune_scratch:
            for glob_key in ["proc??????_save_forward_array.bin",
                             "proc??????_absorb_field.bin"]:
                logger.debug(f"removing '{glob_key}' files from database "
                             f"directory")
                unix.rm(glob(os.path.join(self.model_databases, glob_key)))

    def combine_vol_data_vtk(self, input_path, output_path, hi_res=False,
                             parameters=None):
        """
        Wrapper for 'xcombine_vol_data_vtk'. Combines binary files together
        to generate a single .VTK file that can be visualized by external
        software like ParaView

        .. rubric::
            xcombine_data start end quantity input_dir output_dir hi/lo-res

        .. note::
            It is ASSUMED that this function is being called by
            system.run(single=True) so that we can use the main solver
            directory to perform the kernel summation task

        :type input_path: str
        :param input_path: path to database files to be summed.
        :type output_path: strs
        :param output_path: path to export the outputs of xcombine_sem
        :type hi_res: bool
        :param hi_res: Set the high resolution flag to 1 or True, which will
            generate .vtk files with data at EACH GLL point, rather than at each
            nodal vertex. These files are LARGE, and we discourage using
            `hi_res`==True unless you know you want these files.
        :type parameters: list
        :param parameters: optional list of parameters,
            defaults to `self._parameters`
        """
        unix.cd(self.cwd)

        if parameters is None:
            parameters = self._parameters

        if not os.path.exists(output_path):
            unix.mkdir(output_path)

        # Call on xcombine_sem to combine kernels into a single file
        for name in parameters:
            # e.g.:  bin/xcombine_vol_data_vtk 0 3 alpha_kernel in/ out/ 0
            exc = f"bin/xcombine_vol_data_vtk 0 {self.nproc-1} {name} " \
                  f"{input_path} {output_path} {int(hi_res)}"
            # e.g., smooth_vp.log
            stdout = f"{self._exc2log(exc)}_{name}.log"
            self._run_binary(executable=exc, stdout=stdout, with_mpi=False)

