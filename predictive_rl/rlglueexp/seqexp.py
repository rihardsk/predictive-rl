import subprocess
import atexit
from multiprocessing import Process
import predictive_rl
from predictive_rl.experiment import RLExperiment, rlterminate
import os
import sys


class SequentialExperiment(object):
    def __init__(self):
        self.subprocesses = set()
        self.processes = set()

        @atexit.register
        def kill_subprocesses():
            try:
                print "shuttig down rlglue network.."
                rlterminate()
                print "success!"
            except:
                print "fail!"
            for subproc in self.subprocesses:
                print "killing"
                print subproc
                try:
                    subproc.kill()
                except:
                    pass
            for proc in self.processes:
                print "killing"
                print proc
                try:
                    proc.terminate()
                except:
                    pass

    def run_env(self, rlglue_port, env_file):
        if not os.path.isfile(env_file):
            sys.stderr.write("ERROR: no such env_file: {}".format(env_file))
            return
        env_command = "RLGLUE_PORT={0} java -jar {1}".format(rlglue_port, env_file)
        sys.stderr.write("running:\n")
        sys.stderr.write("\t"+env_command + "\n")
        subproc = subprocess.Popen(env_command, shell=True)
        self.subprocesses.add(subproc)

    def run_rlglue(self, rlglue_port):
        glue_command = "RLGLUE_PORT={0} rl_glue".format(rlglue_port)
        sys.stderr.write("running:\n")
        sys.stderr.write("\t" + glue_command + "\n")
        subproc = subprocess.Popen(glue_command, shell=True)
        self.subprocesses.add(subproc)

    def run_agent(self, rlglue_port, agent_args, agentname):
        os.environ["RLGLUE_PORT"] = str(rlglue_port)
        agentclass = getattr(predictive_rl, agentname)
        agent = agentclass(**agent_args)
        proc = Process(target=agent.run, args=[True], name="Run agent")
        self.processes.add(proc)
        sys.stderr.write("running agent" + "\n")
        proc.start()

    def run_experiment(self, rlglue_port, exp_args):
        os.environ["RLGLUE_PORT"] = str(rlglue_port)
        exp = RLExperiment(**exp_args)
        proc = Process(target=exp.run, args=[True], name="Run experiment")
        self.processes.add(proc)
        sys.stderr.write("running experiment" + "\n")
        proc.start()

    def run(self, rlglue_port, agent_args, exp_args, agent, env_file):
        self.run_env(rlglue_port, env_file)
        self.run_rlglue(rlglue_port)
        self.run_agent(rlglue_port, agent_args, agent)
        self.run_experiment(rlglue_port, exp_args)

        for subproc in self.subprocesses:
            subproc.wait()
        for proc in self.processes:
            proc.join()

    @staticmethod
    def parse_config_file(filename):
        f = open(filename)


def main():
    # rc = Client()
    # dview = rc[:]
    #
    # @dview.parallel(block=True)
    # def run_exp(rlglue_port, env_args, rlglue_args, agent_args, exp_args):
    #     processes = set()
    #     processes.add(subprocess.Popen())
    seqexp = SequentialExperiment()
    seqexp.run(4096, {"action_stdev": 5}, {"dir": "test"}, "PredictiveMockAgent")

if __name__ == "__main__":
    main()



