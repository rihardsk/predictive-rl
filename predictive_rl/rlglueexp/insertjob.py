from jobman import api0, sql
from jobman.tools import flatten
import predictive_rl.rlglueexp.jobexp


def insert_jobexp(exp_args, jobman_args):
    """
    Insert jobman experiment jobs in a sql database.
    Remarks: We have this in a separate module since we can't refer to the jobexp.run function
    from within the jobexp module by it's absolute name (else jobman won't find the experiment when run)
    :param exp_args:
    :param jobman_args:
    :return:
    """
    table_name = jobman_args.get("table_name", "experiment")
    db = api0.open_db('postgres://jobuser:whatever@127.0.0.1/jobbase?table='+table_name)
    for arg in jobman_args:
        sql.insert_job(predictive_rl.rlglueexp.jobexp.run, flatten(arg), db)

if __name__ == "__main__":
    insert_jobexp([(4090, {"agent_args": {"action_stdev": 5}}, {"exp_args": {"dir": "test3"}}),
                   (4091, {"agent_args": {"action_stdev": 1}}, {"exp_args": {"dir": "test3"}})])