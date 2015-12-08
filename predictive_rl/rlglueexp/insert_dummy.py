from jobman import api0, sql
from jobman.tools import DD, flatten
import predictive_rl.rlglueexp.jobexp


def insert_dummy(num):
    base_port = 4030
    table_name = "dummy_exp"
    db = api0.open_db('postgres://jobuser:whatever@127.0.0.1/jobbase?table='+table_name)
    for i in xrange(num):
        state = DD()
        agent_args = DD()
        exp_args = DD()
        agent_args.action_stdev = 1 + i
        exp_args.dir = "dummy_exp"
        state.agent_args = agent_args
        state.exp_args = exp_args
        state.rlglue_port = base_port + i
        sql.insert_job(predictive_rl.rlglueexp.jobexp.run_and_wait_parallel, flatten(state), db)

if __name__ == "__main__":
    insert_dummy(5)