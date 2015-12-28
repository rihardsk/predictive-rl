from ipyparallel import Client

dview = Client()[:]
with dview.sync_imports():
    import predictive_rl.rlglueexp.seqexp


@dview.parallel(block=False)
def _run(args):
    """
    It's assumed that this will be called as _run.map(args)
    """
    import predictive_rl.rlglueexp.seqexp
    print("heeeee")
    seqexp = predictive_rl.rlglueexp.seqexp.SequentialExperiment()
    print("lllll")
    import sys
    sys.stderr.write(str(args) + "\n")
    rlglue_port, agent_args, exp_args, agent = args
    # rlglue_port, agent_args, exp_args = (1,2,3)
    print("ooooo")
    res = seqexp.run(rlglue_port, agent_args, exp_args, agent)
    return rlglue_port


def run(args):
    return _run.map(args)


def main():
    # parser = argparse.ArgumentParser(description="Run parallel experiments using ipython as defined in a config file.")
    run([(4090, {"action_stdev": 5}, {"dir": "test3"}), (4091, {"action_stdev": 1}, {"dir": "test3"})])


if __name__ == "__main__":
    main()
