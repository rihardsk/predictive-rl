from ipyparallel import Client
import subprocess
import shlex
import atexit
from multiprocessing import Process
from predictive_rl.predictive_future_agent import PredictiveFutureAgent as pfe
from predictive_rl.experiment import RLExperiment
import os


def rlglue_experiment(rlglue_port, agent_args, exp_args):
    subprocesses = set()
    processes = set()

    @atexit.register
    def kill_subprocesses():
        for subproc in subprocesses:
            print "killing"
            print subproc
            subproc.kill()
        for proc in processes:
            print "killing"
            print proc
            proc.terminate()

    env_command = "RLGLUE_PORT={0} java -jar /home/rihards/Programming/Multi/rl/rl-library/rl-library/products/CartPole.jar".format(rlglue_port)
    glue_command = "RLGLUE_PORT={0} rl_glue".format(rlglue_port)
    commands = [env_command, glue_command]
    for command in commands:
        subprocesses.add(subprocess.Popen(command, shell=True))
    agent = pfe(**agent_args)
    exp = RLExperiment(**exp_args)
    os.environ["RLGLUE_PORT"] = str(rlglue_port)
    processes.add(Process(target=agent.run))
    processes.add(Process(target=exp.run))

    for proc in processes:
        proc.start()
    for subproc in subprocesses:
        subproc.wait()
    for proc in processes:
        proc.join()


def main():
    # rc = Client()
    # dview = rc[:]
    #
    # @dview.parallel(block=True)
    # def run_exp(rlglue_port, env_args, rlglue_args, agent_args, exp_args):
    #     processes = set()
    #     processes.add(subprocess.Popen())

    rlglue_experiment(4096, {"action_stdev": 5}, {"dir": "test"})

if __name__ == "__main__":
    main()



