from ipyparallel import Client
import predictive_rl.rlglueexp.seqexp


class IPythonExperiment(object):
    dview = Client()[:]
    with dview.sync_imports():
        import predictive_rl.rlglueexp.seqexp


    @staticmethod
    @dview.parallel
    def run(args):
        seqexp = predictive_rl.rlglueexp.seqexp.SequentialExperiment()
        rlglue_port, agent_args, exp_args = args
        return seqexp.run_and_wait(rlglue_port, agent_args, exp_args)


def main():
    # parser = argparse.ArgumentParser(description="Run parallel experiments using ipython as defined in a config file.")
    ipyexp = IPythonExperiment()
    ipyexp.run([(4090, {"action_stdev": 5}, {"dir": "test3"}), (4091, {"action_stdev": 1}, {"dir": "test3"})])


if __name__ == "__main__":
    main()
