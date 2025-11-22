import os
# import git
# import subprocess
import platform
import random
import sys

import numpy as np
import torch


def reproducibility():
    """
    Set the ennvironment variables and configurations so subsequent runnings
    of the experiments can be reproduced.
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_machine_info():
    """
    Return the information about the OS, platform, machine's name on the
    current network and the processor.
    """

    system = platform.system()
    plf = platform.platform(aliased=True)
    node = platform.node()
    processor = platform.processor()

    return dict(system=system, platform=plf, node=node, processor=processor)


def get_configurations(args):
    """
    Get the configuration of a running from the program's arguments. 

    Modify the attributes of `config` according to the
    values of `args`.  The attributes added to `config` are:

    * `output_dir`: directory name where the directory structure of the
      results will be created.
    * `gpu_id`: The identifier of the GPU that is used to run the
      experiments. 
    * `yaml_paths`: list of file paths in YAML format where the parameters of
      the experiments are stored. See `utils.io.read_yaml` to see which
      params are those.
    * `batch_size`: 
    * `cmd`: string of alternative parameters for the experiments. See
        for more information.
    * `environ_variables`: all the OS environments in the session used to run
      the program.
    * `machine`: Information about the machine that runs the experiments.
      See `utils.configuration.get_machine_info()` for more information.
    """

    cmd_argv = " ".join((["python"] + sys.argv))

    d = {
        "output_dir": args.output_dir,
        "gpu_id": args.gpu,
        "yaml_paths": args.param,
        "cmd": cmd_argv,
        "environ_variables": dict(os.environ),
        "machine": get_machine_info()
    }
    
    if args.batch_size:
        d["batch_size"] = args.batch_size

    # git hash
    # git_hash = git.cmd.Git("./").rev_parse("HEAD")
    # setattr(config, "git_hash", git_hash)

    # # branch
    # _cmd = "git rev-parse --abbrev-ref HEAD"
    # branch = subprocess.check_output(_cmd.split()).strip().decode("utf-8")
    # branch = "-".join(branch.split("/"))
    # setattr(config, "branch", branch)
    return d


def correct_param(param, param_initialpoint, param_stepsize, dataset):
    param_stepsize["max_iter"] = param["max_iter"]
    param_stepsize["epsilon"] = param["epsilon"]
    param_initialpoint["epsilon"] = param["epsilon"]
    param_initialpoint["dataset"] = dataset
