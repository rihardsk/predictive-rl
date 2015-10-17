#!/usr/bin/env python
"""
This is an RLGlue experiment designed to collect the type of data
presented in:

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

(Based on the sample_experiment.py from the Rl-glue python codec examples.)

usage: experiment.py [-h] [--num_epochs NUM_EPOCHS]
                          [--epoch_length EPOCH_LENGTH]
                          [--test_length TEST_LENGTH]

Author: Nathan Sprague

"""
import rlglue.RLGlue as RLGlue
import argparse
import os
import time
import numpy as np


def run_epoch(epoch, num_steps, prefix, collect_reward=False):
    """ Run one 'epoch' of training or testing, where an epoch is defined
    by the number of steps executed.  Prints a progress report after
    every trial

    Arguments:
       num_steps - steps per epoch
       prefix - string to print ('training' or 'testing')

    """
    steps_left = num_steps
    if prefix == "training" or not collect_reward:
        while steps_left > 0:
            print prefix + " epoch: ", epoch, "steps_left: ", steps_left
            terminal = RLGlue.RL_episode(steps_left)
            if not terminal:
                RLGlue.RL_agent_message("episode_end")
            steps_left -= RLGlue.RL_num_steps()
    elif prefix == "testing":
        total_reward = 0
        episode_counter = 0
        terminal = False
        while steps_left > 0:
            if terminal:
                print prefix + " epoch: ", epoch, "steps_left: ", steps_left
            roat = RLGlue.RL_step()
            reward = roat.r
            terminal = roat.terminal
            total_reward += reward
            episode_counter += terminal
            steps_left -= 1
        return total_reward, episode_counter


def update_results_file(epoch, total_reward, num_episodes, results_file):
    out = "{},{},{},{}\n".format(epoch, num_episodes, total_reward,
                                 total_reward / float(num_episodes))
    results_file.write(out)


def open_results_file(exp_dir):
    print "OPENING ", exp_dir + '/results.csv'
    results_file = open(os.path.join(exp_dir, 'results.csv'), 'w', 0)
    results_file.write('epoch,num_episodes,total_reward,reward_per_epoch\n')
    return results_file


def main():
    """
    Run the desired number of training epochs, a testing epoch
    is conducted after each training epoch.
    """

    parser = argparse.ArgumentParser(description='Neural rl experiment.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--epoch_length', type=int, default=50000,
                        help='Number of steps per epoch')
    parser.add_argument('--test_length', type=int, default=10000,
                        help='Number of steps per test')
    parser.add_argument('--dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--agent_suffix', type=str, default='',
                        help='Agent specific suffix to append to the results dir name')
    parser.add_argument('--collect_rewards', type=bool, default=False,
                        help='If set to true, testing episode mean rewards will be saved to a file. \
                              In general this leads to testing epochs being longer than usual.')

    args = parser.parse_args()

    time_str = time.strftime("%m-%d-%H-%M_", time.gmtime())
    experiment_dir = os.path.join(args.dir, time_str + args.agent_suffix)

    try:
        os.stat(experiment_dir)
    except:
        os.makedirs(experiment_dir)

    results_file = open_results_file(experiment_dir)
    RLGlue.RL_init()
    RLGlue.RL_agent_message("set_dir " + experiment_dir)

    for epoch in range(1, args.num_epochs + 1):
        RLGlue.RL_agent_message("start_epoch " + str(epoch))
        # curtime = time.time()
        run_epoch(epoch, args.epoch_length, "training")
        # duration = time.time() - curtime
        # print "training epoch " + str(epoch) + " " + str(duration)
        RLGlue.RL_agent_message("finish_epoch " + str(epoch))

        if args.test_length > 0:
            RLGlue.RL_agent_message("start_testing")
            # curtime = time.time()
            if args.collect_rewards:
                total_reward, num_episodes = run_epoch(epoch, args.test_length, "testing", True)
                # TODO save the rewards here
            else:
                run_epoch(epoch, args.test_length, "testing")
            # duration = time.time() - curtime
            # print "testing epoch " + str(epoch) + " " + str(duration)
            RLGlue.RL_agent_message("finish_testing " + str(epoch))

if __name__ == "__main__":
    main()
