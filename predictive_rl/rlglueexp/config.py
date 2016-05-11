from configobj import ConfigObj
from multiprocessing import Process
import multiprocessing
import logging
import os


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
    import insertjob
    return insertjob.insert_jobexp(*parse(configfilename))


def run_ipyexp(configfilename):
    res = run_ipyexp_async(configfilename)
    res.wait()
    return res.result


def run_ipyexp_async(configfilename):
    import predictive_rl.rlglueexp.ipyexp
    expargs, jobargs = parse(configfilename)
    ipyargs = [(v["rlglue_port"], v["agent_args"], v["exp_args"],
                v["agent"], v["env_file"]) for v in expargs]
    res = predictive_rl.rlglueexp.ipyexp.run(ipyargs)
    return res


def run_seqexp(configfilename):
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
    # res = parse("sample.config")
    multiprocessing.log_to_stderr(logging.DEBUG)
    # res = run_ipyexp("sample.config")
    # res = run_seqexp("../../experiments/short.config")
    res = run_ipyexp("../../experiments/short.config")
    temp = 0
