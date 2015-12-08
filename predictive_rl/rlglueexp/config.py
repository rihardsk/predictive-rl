from configobj import ConfigObj
import insertjob
from ipyexp import IPythonExperiment


def parse(configfilename):
    config = ConfigObj(configfilename)
    starting_port = config.get("starting_port")

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
    exp = IPythonExperiment()
    expargs, jobargs = parse(configfilename)
    ipyargs = [(v["rlglue_port"], v["agent_args"], v["exp_args"]) for v in expargs]
    exp.run(ipyargs)


if __name__ == "__main__":
    res = parse("sample.config")
    temp = 0
