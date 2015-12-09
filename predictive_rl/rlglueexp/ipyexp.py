from ipyparallel import Client

dview = Client()[:]
with dview.sync_imports():
    import predictive_rl.rlglueexp.seqexp


@dview.parallel(block=False)
def run(args):
    import predictive_rl.rlglueexp.seqexp
    print("heeeee")
    seqexp = predictive_rl.rlglueexp.seqexp.SequentialExperiment()
    print("lllll")
    import sys
    sys.stderr.write(str(args) + "\n")
    rlglue_port, agent_args, exp_args = args[0]
    # rlglue_port, agent_args, exp_args = (1,2,3)
    print("ooooo")
    res = seqexp.run_and_wait(rlglue_port, agent_args, exp_args)
    return rlglue_port


def main():
    # parser = argparse.ArgumentParser(description="Run parallel experiments using ipython as defined in a config file.")
    run([(4090, {"action_stdev": 5}, {"dir": "test3"}), (4091, {"action_stdev": 1}, {"dir": "test3"})])


if __name__ == "__main__":
    main()
