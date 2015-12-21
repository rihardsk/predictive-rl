from configobj import ConfigObj
import insertjob
from multiprocessing import Process
import multiprocessing
import logging


def parse(configfilename):
    config = ConfigObj(configfilename)
    starting_port = config.get("starting_port")
    if starting_port is not None:
        starting_port = int(starting_port)

    def getexpargs():
        for i, sname in enumerate(config.sections):
            section = config[sname]
            if section.get("rlglue_port") is None:
                section["rlglue_port"] = starting_port + i
            yield section

    sectionset = set(config.sections)
    jobargs = {k: v for k, v in config.iteritems() if k not in sectionset}

    return list(getexpargs()), jobargs


def run_jobexp(configfilename):
    return insertjob.insert_jobexp(*parse(configfilename))


def run_ipyexp(configfilename):
    res = run_ipyexp_async(configfilename)
    res.wait()
    return res.result


def run_ipyexp_async(configfilename):
    import predictive_rl.rlglueexp.ipyexp
    expargs, jobargs = parse(configfilename)
    ipyargs = [(v["rlglue_port"], v["agent_args"], v["exp_args"], v["agent"]) for v in expargs]
    res = predictive_rl.rlglueexp.ipyexp.run(ipyargs)
    return res


def run_seqexp(configfilename):
    from predictive_rl.rlglueexp.seqexp import SequentialExperiment
    expargs, jobargs = parse(configfilename)
    processes = set()
    exps = set()
    for arg in expargs:
        exp = SequentialExperiment()
        proc = Process(target=exp.run, kwargs=arg)
        proc.start()
        proc.join()



if __name__ == "__main__":
    # res = parse("sample.config")
    multiprocessing.log_to_stderr(logging.DEBUG)
    # res = run_ipyexp("sample.config")
    res = run_seqexp("../../experiments/short.config")
    temp = 0
