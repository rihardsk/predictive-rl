import predictive_rl.rlglueexp.seqexp


def run_and_wait_parallel(state, channel):
    rlglue_port = state.rlglue_port
    agent_args = state.agent_args
    exp_args = state.exp_args
    seqexp = predictive_rl.rlglueexp.seqexp.SequentialExperiment()
    seqexp.run_and_wait(rlglue_port, agent_args, exp_args)


