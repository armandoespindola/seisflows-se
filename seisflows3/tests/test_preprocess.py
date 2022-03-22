"""
Test suite for the SeisFlows3 system module, which controls interaction with
various compute systems
"""
import os
import sys
import shutil
import pytest
from unittest.mock import patch
from seisflows3 import config
from seisflows3.seisflows import SeisFlows, return_modules


# The module that we're testing, allows for copy-pasting these test suites
MODULE = "preprocess"

# Ensures that these parameters are always defined, even when using subclasses
REQUIRED_PARAMETERS = []
REQUIRED_FUNCTIONS = ["required", "check", "setup", "prepare_eval_grad",
                      "sum_residuals", "finalize"]

# Define some re-used paths
TEST_DIR = os.path.join(config.ROOT_DIR, "tests")
REPO_DIR = os.path.abspath(os.path.join(config.ROOT_DIR, ".."))


@pytest.fixture
def copy_par_file(tmpdir):
    """
    Copy the template parameter file into the temporary test directory
    :rtype: str
    :return: location of the parameter file
    """
    src = os.path.join(TEST_DIR, "test_data", "test_filled_parameters.yaml")
    dst = os.path.join(tmpdir, "parameters.yaml")
    shutil.copy(src, dst)


@pytest.fixture
def modules():
    """
    Return a list of subclasses that falls under the System module
    """
    return return_modules()[MODULE]


@pytest.fixture
def sfinit(tmpdir, copy_par_file):
    """
    Re-used function that will initate a SeisFlows3 working environment in
    sys modules
    :return:
    """
    copy_par_file
    os.chdir(tmpdir)
    with patch.object(sys, "argv", ["seisflows"]):
        sf = SeisFlows()
        sf._register(force=True)
    config.init_seisflows(check=False)

    return sf


def test_import(sfinit, modules):
    """
    Test code by importing all available classes for this module.
    If any of these fails then the module itself has some code error
    (e.g., syntax errors, inheritance errors).
    """
    sfinit
    for package, module_list in modules.items():
        for module in module_list:
            config.custom_import(MODULE, module)()


def test_validate(sfinit, modules):
    """
    Test out path and parameter validation, essentially checking that all
    the paths and parameters are set properly

    .. note::
        This doesn't work because we have required parameters that are not
        set in the default parameter file. We can run configure beforehand
        but does that make sense?
    :return:
    """
    return
    sfinit
    for package, module_list in modules.items():
        for module in module_list:
            loaded_module = config.custom_import(MODULE, module)()
            from IPython import embed;embed()
            loaded_module.required.validate()


def test_required_parameters_exist(sfinit, modules):
    """
    Ensure that the required parameters are set in all the classes/subclasses
    That is, that the parameters defined above in REQUIRED_PARAMETERS have been
    defined by each SYSTEM class
    """
    sfinit
    for package, module_list in modules.items():
        for module in module_list:
            loaded_module = config.custom_import(MODULE, module)()
            sf_pp = loaded_module.required
            # Check that required parameters are set
            for req_par in REQUIRED_PARAMETERS:
                assert(req_par in sf_pp.parameters.keys()), \
                    f"{req_par} is a required parameter for module {MODULE}"


def test_required_functions_exist(sfinit, modules):
    """
    Make sure that the named, required functions exist within the class
    Do not execute just make sure they're defined, because they will be
    expected by other modules
    """
    sfinit
    for package, module_list in modules.items():
        for module in module_list:
            loaded_module = config.custom_import(MODULE, module)()
            for func in REQUIRED_FUNCTIONS:
                assert(func in dir(loaded_module)), \
                    f"'{func}' is a required function in module: " \
                    f"{MODULE}.{module}"


# ==============================================================================
# MODULE AND FUNCTION SPECIFIC TESTS TO FOLLOW
# ==============================================================================
def test_default_check(sfinit):
    """
    Test seisflows3.preprocess.default.check()

    :param sfinit:
    :param modules:
    :return:
    """
    sfinit
    PAR = sys.modules["seisflows_parameters"]
    preprocess = sys.modules["seisflows_preprocess"]

    # Make sure that the check statement catches incorrectly set parameters
    incorrect_parameters = {
        "NORMALIZE": ["JNORML3"],
        "MUTE": ["not_mute"],
        "FILTER": "bondpass",
        "FORMAT": "not_an_acceptable_format"
    }
    for key, val in incorrect_parameters.items():
        og_val = PAR[key]
        with pytest.raises(AssertionError):
            PAR.force_set(key, val)
            preprocess.check()
        PAR.force_set(key, og_val)

    # Make sure that parameters set to inappropriate values throw assertions
    correct_parameters = {
        "FILTER": "BANDPASS",
        "WORKFLOW": "INVERSION",
        "MIN_FREQ": 1,
    }
    for key, val in correct_parameters.items():
        PAR.force_set(key, val)

    incorrect_values = {
        "MAX_FREQ": -1,
    }
    for key, val in incorrect_values.items():
        og_val = PAR[key]
        with pytest.raises(AssertionError):
            PAR.force_set(key, val)
            preprocess.check()
        PAR.force_set(key, og_val)


def test_default_setup(sfinit):
    """
    Ensure that default setup correctly sets up the preprocessing machinery
    """
    sfinit
    PAR = sys.modules["seisflows_parameters"]
    preprocess = sys.modules["seisflows_preprocess"]

    # Make sure that preprocess machinery is set up empty
    assert(preprocess.misfit is None)
    assert(preprocess.adjoint is None)
    assert(preprocess.reader is None)
    assert(preprocess.writer is None)

    # Set some default parameters to run setup
    misfit_name = "waveform"
    io_name = "ascii"
    PAR.force_set("MISFIT", misfit_name)
    PAR.force_set("FORMAT", io_name)
    preprocess.setup()
    pytest.set_trace()

    assert(preprocess.misfit.__name__ == misfit_name)
    assert(preprocess.adjoint.__name__ == misfit_name)
    assert(preprocess.reader.__name__ == io_name)
    assert(preprocess.writer.__name__ == io_name)


# def test_default_prepare_eval_grad(tmpdir, sfinit):
#     """
#     Ensure that prepare_eval_grad writes out adjoint traces and auxiliary files
#     """
#     sfinit
#     PAR = sys.modules["seisflows_parameters"]
#     preprocess = sys.modules["seisflows_preprocess"]
#
#     cwd = tmpdir
#     taskid = 0
#     filenames = []
#     preprocess.prepare_eval_grad(cwd=cwd, taskid=taskid, filenames=filenames)
#     pytest.set_trace()
