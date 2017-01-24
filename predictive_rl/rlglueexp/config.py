from configobj import ConfigObj
from multiprocessing import Process
import multiprocessing
import logging
import os
import argparse


def parse(configfilename):
    config = ConfigObj(configfilename)
    starting_port = config.get("starting_port")
    env_base_path = os.environ.get("ENV_BASE_PATH") or config.get("env_base_path")
    if starting_port is not None:
        starting_port = int(starting_port)

    def getexpargs():
        for i, sname in enumerate(config.sections):
            section = config[sname]
            if section.get("rlglue_port") is None:
                section["rlglue_port"] = starting_port + i
            if env_base_path is not None and section["env_file"] is not None:
                section["env_file"] = os.path.join(env_base_path, section["env_file"])
            yield section

    sectionset = set(config.sections)
    jobargs = {k: v for k, v in config.iteritems() if k not in sectionset}

    return list(getexpargs()), jobargs


def run_jobexp(configfilename):
    """
    Runs an experiment using jobman. Currently it is probably broken though.
    """
    import insertjob
    return insertjob.insert_jobexp(*parse(configfilename))


def run_ipyexp(configfilename):
    """
    Runs the experiments on an IPython cluster. This allows them to be run
    in parallel and/or on a remote machine. See
    `IPython Parallel docs http://ipyparallel.readthedocs.io`_

    :param configfilename: a file containing the experiment specification
    :returns: the result is not meaningful. A value is returned only to
    indicate that the experiment has finished.
    """
    res = run_ipyexp_async(configfilename)
    res.wait()
    return res.result


def run_ipyexp_async(configfilename):
    """
    Runs the experiments on an IPython cluster. This allows them to be run
    in parallel and/or on a remote machine. See
    `IPython Parallel docs http://ipyparallel.readthedocs.io`_

    :param configfilename: a file containing the experiment specification
    :returns: an IPython AsuncResult object that can be used to query
    the job state and retrieve the result. See
    `IPython Parallel docs http://ipyparallel.readthedocs.io/en/latest/asyncresult.html`_
    """
    import predictive_rl.rlglueexp.ipyexp
    expargs, jobargs = parse(configfilename)
    ipyargs = [(v["rlglue_port"], v["agent_args"], v["exp_args"],
                v["agent"], v["env_file"]) for v in expargs]
    res = predictive_rl.rlglueexp.ipyexp.run(ipyargs)
    return res


def run_seqexp(configfilename):
    """
    Starts the whole set of processes necessary for the RL experiment
    according to an experiment configuration file. This includes rlglue,
    the environment, the agent and the experiment.

    :param configfilename: a file containing the experiment specification
    """
    from predictive_rl.rlglueexp.seqexp import SequentialExperiment
    expargs, jobargs = parse(configfilename)
    processes = set()
    exps = set()
    for arg in expargs:
        exp = SequentialExperiment()
        # proc = Process(target=exp.run, kwargs=arg, name="Run seqexp")
        # proc.start()
        # proc.join()
        exp.run(**arg)


if __name__ == "__main__":
    experiment_types = ["ipython", "jobman", "local"]
    parser = argparse.ArgumentParser(
        description='Automate an experiment according to a experiment \
        configuration file. It can be run on the "ipython", "jobman", \
        or "local" backend.\nWARNING: "jobman" backend is probably broken.')
    parser.add_argument('config_file', type=str,
                        help="a file with the experiments specification. \
                        See `sample.config`_.")
    parser.add_argument('-b', '--backend', choices=experiment_types,
                        default="local", help="the backend to run the experiment.")
    args = parser.parse_args()

    backends = {
        "ipython": run_ipyexp,
        "jobman": run_jobexp,
        "local": run_seqexp,
    }

    method = backends[args.backend]
    method(args.config_file)
