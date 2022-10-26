import os
import time

from rl_trainer import RL_Trainer
from trading_agent import TradingAgent

def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=40)
    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)


    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    #################################
    ### CREATE DIRECTORY FOR LOGGING
    #################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    agent = Trading_Agent(env, params)
    # TODO - do the training here (we might want to keep it all in trading_agent.py and scrap rl_trainer altogether)
    #agent.rl_trainer.run_training_loop()


if __name__ == "__main__":
    main()
