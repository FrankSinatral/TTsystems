import argparse
import numpy as np

def parse_list(string):
    # turn a string into a list
    return [float(x) if x != 'None' else None for x in string.split(',')]

def get_config():
    
    parser = argparse.ArgumentParser(
        description="rl_training_tt", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ## ------------------------------ basic setting ----------------------------------- ##
    parser.add_argument(
        "--algo_name",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="the agent that you choose to train"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="two_trailer",
        choices=["standard_parking", "single_tractor", "one_trailer", "two_trailer", "three_trailer"],
        help="which env you are using"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for numpy/torch"
    )
    ## ------------------------------ env setting --------------------------------- ##
    parser.add_argument(
        "--reward_type",
        '-r',
        type=str,
        # it seems that diff_distance is the best now
        default="diff_distance",
        choices=["diff_distance", "parking_reward", "potential_reward",
                 "penalty_reward", "penalty_potential"],
        help="the type of reward used")
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="whether evaluate mode for env"
    )
    parser.add_argument(
        "--allow_backward",
        action='store_true',
        help="whether to allow backward in model step"
    )
    parser.add_argument(
        '--distance_weights', 
        nargs='*', 
        type=float, 
        default=[1.00, 1.00, 1.00, 1.00, 1.00], 
        help='List of distance matrix values'
    )
    parser.add_argument(
        '--reward_weights', 
        nargs='*', 
        type=float, 
        default=[1, 0.3, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 
        help='reward weights used in parking reward'
    )
    parser.add_argument(
        "--goal",
        type=parse_list,
        default=[0, 0, 0, 0, 0, None],
        help="goal position"
    )
    parser.add_argument(
        '--xmax',
        type=float,
        default=10.0,
        help='the initial x-axis range'
    )
    parser.add_argument(
        '--ymax',
        type=float,
        default=10.0,
        help='the initial y-axis range'
    )
    parser.add_argument(
        '--max_episode_steps',
        type=int,
        default=300,
        help='max episode steps of the env'
    )
    parser.add_argument(
        '--penalty_backward',
        type=float,
        default=2.0,
        help='penalty for backward'
    )
    parser.add_argument(
        '--penalty_switch',
        type=float,
        default=10.0,
        help='penalty for switch'
    )
    parser.add_argument(
        '--edge',
        type=float,
        default=15.0,
        help='map edge for vis'
    )
    parser.add_argument(
        '--constraint_coeff',
        type=float,
        default=2.0,
        help='terminated condition'
    )
    parser.add_argument(
        "--sucess_goal_reward_parking",
        type=float,
        default=-0.12,
        help="success goal reward for parking"
    )
    parser.add_argument(
        "--sucess_goal_reward_others",
        type=float,
        default=100,
        help="success goal reward others"
    )
    ## ------------------------------- C_three_trailer setting ----------------------------------- ##
    ### --------------------------------- basic setting --------------------------------------------------- ###
    parser.add_argument("--pi", type=float, default=np.pi, help="Value of PI")
    parser.add_argument("--xi_max", type=float, default=(np.pi) / 4, help="Maximum steering angle")
    parser.add_argument("--xy_reso", type=float, default=2.0, help="[m] grid resolution")
    parser.add_argument("--yaw_reso", type=float, default=np.deg2rad(15.0), help="[rad] yaw resolution")
    parser.add_argument("--goal_yaw_error", type=float, default=np.deg2rad(3.0), help="[rad] allowable yaw error for goal")
    parser.add_argument("--max_steer", type=float, default=0.6, help="[rad] maximum steering angle")
    parser.add_argument("--v_max", type=float, default=2.0, help="Maximum velocity")
    ### --------------------------------- MP setting ---------------------------------------------------- ###
    parser.add_argument("--move_step", type=float, default=0.2, help="[m] path interporate resolution")
    parser.add_argument("--n_steer", type=float, default=20.0, help="number of steer command")
    
    ### --------------------------------- collision_check setting ------------------------------------------- ###
    parser.add_argument("--collision_check_step", type=int, default=10, help="skip number for collision check")
    parser.add_argument("--extend_area", type=float, default=0.0, help="[m] map extend length")
    parser.add_argument(
        "--safe_d",
        type=float,
        default=0.0,
        help="the safe distance from the vehicle to obstacle"
    )
    
    ### ---------------------------------- heuristic setting (hyperparameters) ------------------------------ ###
    parser.add_argument("--gear_cost", type=float, default=100.0, help="switch back penalty cost")
    parser.add_argument("--backward_cost", type=float, default=5.0, help="backward penalty cost")
    parser.add_argument("--steer_change_cost", type=float, default=5.0, help="steer angle change penalty cost")
    parser.add_argument("--steer_angle_cost", type=float, default=1.0, help="steer angle penalty cost")
    parser.add_argument("--scissors_cost", type=float, default=2.0, help="scissors movement penalty cost")
    parser.add_argument("--h_cost", type=float, default=1.0, help="Heuristic cost")
    ### ----------------------------------- model setting ------------------------------------------------ ###
    parser.add_argument("--w", type=float, default=2.0, help="[m] width of vehicle")
    parser.add_argument("--wb", type=float, default=3.5, help="[m] wheel base: rear to front steer")
    parser.add_argument("--wd", type=float, default=1.4, help="[m] distance between left-right wheels (0.7 * W)")
    parser.add_argument("--rf", type=float, default=4.5, help="[m] distance from rear to vehicle front end")
    parser.add_argument("--rb", type=float, default=1.0, help="[m] distance from rear to vehicle back end")
    parser.add_argument("--tr", type=float, default=0.5, help="[m] tyre radius")
    parser.add_argument("--tw", type=float, default=1.0, help="[m] tyre width")
    ## ---------------------------------- model setting / trailer setting ------------------------------- ##
    parser.add_argument("--rtr", type=float, default=2.0, help="[m] rear to trailer wheel")
    parser.add_argument("--rtf", type=float, default=1.0, help="[m] distance from rear to trailer front end")
    parser.add_argument("--rtb", type=float, default=3.0, help="[m] distance from rear to trailer back end")
    parser.add_argument("--rtr2", type=float, default=2.0, help="[m] rear to second trailer wheel")
    parser.add_argument("--rtf2", type=float, default=1.0, help="[m] distance from rear to second trailer front end")
    parser.add_argument("--rtb2", type=float, default=3.0, help="[m] distance from rear to second trailer back end")
    parser.add_argument("--rtr3", type=float, default=2.0, help="[m] rear to third trailer wheel")
    parser.add_argument("--rtf3", type=float, default=1.0, help="[m] distance from rear to third trailer front end")
    parser.add_argument("--rtb3", type=float, default=3.0, help="[m] distance from rear to third trailer back end")
    
    ## ------------------------------- file parts -------------------------------------- ##
    parser.add_argument(
        "--logging_dir",
        type=str,
        default='rl_training/logging/',
        help="logging saving directory"
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        default='my_progress.txt', # set to results.txt for evaluation
        help="logging information file"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs_rl/",
        help="file directory where to tensorboard"
    )
    parser.add_argument(
        "--save_freq",
        type=str, 
        default=10,
        help="save freq of your model"
    )
    ## ----------------------------- Actor Critic Setting ----------------------------- ##
    parser.add_argument(
        "--hidden_sizes",
        nargs='+',
        type=int,
        default=[512, 512, 512],
        help="Sizes of the hidden layers in the MLPActorCritic"
    )
    parser.add_argument(
        '--activation',
        type=str,
        choices=["ReLU","Tanh"],
        default='ReLU',
        help="Activation function for the MLPActorCritic")
    ## ----------------------------- hyperparameters setting -------------------------- ##
    
    
    parser.add_argument(
        "--num_test_episodes",
        type=str,
        default=10,
        help="how many rollouts in evaluate step"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        # 0.99 for default ppo
        help="gamma for MDP"
    )
    
    ## ---------------------------- sac setting ----------------------------------- ##
    parser.add_argument(
        "--sac_epochs",
        type=int,
        default=12500,
        help="how many epochs to update"
    )
    parser.add_argument(
        "--sac_steps_per_epoch",
        type=int,
        default=4,
        help="steps in each epoch"
    )
    
    parser.add_argument(
        "--replay_size",
        type=int,
        default=int(1e6),
        help="replay buffer size"
    )
    
    parser.add_argument(
        "--polyak",
        type=float,
        default=0.95,
        help="the weight between two nn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate of sac"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="alpha"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="mini batch for gradient descent step"
    )
    parser.add_argument(
        "--start_steps",
        type=int,
        default=100,
        help="start random action number"
    )
    parser.add_argument(
        "--update_after",
        type=int,
        default=100,
        help="how many steps will you start to update"
    )
    parser.add_argument(
        "--update_every",
        type=int,
        default=1,
        help="how many time to update"
    )
    parser.add_argument(
        "--whether_her",
        action='store_true',
        help="whether to apply her buffer"
    )
    parser.add_argument(
        "--use_auto",
        action='store_true',
        help="automatically tuning alpha"
    )
    ## ------------------------------------ ppo setting ------------------------------------- ##
    parser.add_argument(
       "--clip_ratio",
       type=float,
       default=0.2,
       help="clip ratio for ppo actor loss" 
    )
    parser.add_argument(
        "--pi_lr",
        type=float,
        default=3e-4,
        help="actor learning rate"
    )
    parser.add_argument(
        "--vf_lr",
        type=float,
        default=1e-3,
        help="value function learning rate"
    )
    parser.add_argument(
        "--train_pi_iters",
        type=int,
        default=80,
        help="how many updates for pi nn"
    )
    parser.add_argument(
        "--train_v_iters",
        type=int,
        default=80,
        help="how many updates for value nn"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.97,
        help="lambda setting"
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.1,
        help="early stop using target kl"
    )
    parser.add_argument(
        "--local_steps_per_epoch",
        type=int,
        default=4000,
        help="temp buffer for ppo"
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=50,
        help="how many epochs to update"
    )
    ## ------------------------------------ evaluation setting ------------------------------------ ##
    parser.add_argument(
        '--len', 
        '-l', 
        type=int, 
        default=0)
    parser.add_argument(
        '--episodes', 
        '-n', 
        type=int, 
        default=10)
    parser.add_argument(
        '--norender', 
        '-nr', 
        action='store_true')
    parser.add_argument(
        '--itr', 
        '-i', 
        type=int, 
        default=-1)
    parser.add_argument(
        '--deterministic', 
        '-d', 
        action='store_true')
    parser.add_argument(
        "--evaluate_dir",
        type=str,
        default='rl_training/evaluation/',
        help="logging saving directory"
    )
    return parser
    