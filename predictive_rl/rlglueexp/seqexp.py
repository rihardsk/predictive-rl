import subprocess
import atexit
from multiprocessing import Process
from predictive_rl.predictive_future_agent import PredictiveFutureAgent as pfe
from predictive_rl.experiment import RLExperiment
import os
import sys


class SequentialExperiment(object):
    def __init__(self):
        self.subprocesses = set()
        self.processes = set()

        @atexit.register
        def kill_subprocesses():
            for subproc in self.subprocesses:
                print "killing"
                print subproc
                subproc.kill()
            for proc in self.processes:
                print "killing"
                print proc
                proc.terminate()

    def run_env(self, rlglue_port):
        env_command = "RLGLUE_PORT={0} java -jar /home/rihards/Programming/Multi/rl/rl-library/rl-library/products/CartPole.jar".format(rlglue_port)
        sys.stderr.write("running:\n")
        sys.stderr.write("\t"+env_command + "\n")
        self.subprocesses.add(subprocess.Popen(env_command, shell=True))

    def run_rlglue(self, rlglue_port):
        glue_command = "RLGLUE_PORT={0} rl_glue".format(rlglue_port)
        sys.stderr.write("running:\n")
        sys.stderr.write("\t" + glue_command + "\n")
        self.subprocesses.add(subprocess.Popen(glue_command, shell=True))

    def run_agent(self, rlglue_port, agent_args):
        os.environ["RLGLUE_PORT"] = str(rlglue_port)
        agent = pfe(**agent_args)
        proc = Process(target=agent.run, args=[True])
        self.processes.add(proc)
        sys.stderr.write("running agent" + "\n")
        proc.start()

    def run_experiment(self, rlglue_port, exp_args):
        os.environ["RLGLUE_PORT"] = str(rlglue_port)
        exp = RLExperiment(**exp_args)
        proc = Process(target=exp.run, args=[True])
        self.processes.add(proc)
        sys.stderr.write("running experiment" + "\n")
        proc.start()

    def run(self, rlglue_port, agent_args, exp_args):
        self.run_env(rlglue_port)
        self.run_rlglue(rlglue_port)
        self.run_agent(rlglue_port, agent_args)
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
    seqexp.run(4096, {"action_stdev": 5}, {"dir": "test"})

if __name__ == "__main__":
    main()



