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
        "--vehicle_type",
        type=str,
        default="one_trailer",
        choices=["standard_parking", "single_tractor", "one_trailer", "two_trailer", "three_trailer"],
        help="which env you are using"
    )
    parser.add_argument(
        "--seed", type=int, default=24, help="random seed for numpy/torch"
    )
    
    parser.add_argument(
        "--use_stable_baseline",
        default=True,
        action='store_true',
        help="whether use stable baseline"
    )
    
    ## ------------------------------ our env setting --------------------------------- ##
    parser.add_argument(
        "--verbose",
        default=True,
        action='store_true',
        help="print config setting"
    )
    
    parser.add_argument(
        "--reward_type",
        '-r',
        type=str,
        # it seems that diff_distance is the best now
        default="potential_reward",
        choices=["diff_distance", "parking_reward", "potential_reward"],
        help="the type of reward used")
    parser.add_argument(
        "--evaluate_mode",
        action="store_true",
        default=True,
        help="whether evaluate mode for env"
    )
    parser.add_argument(
        "--allow_backward",
        action='store_true',
        help="whether to allow backward in model step"
    )
    # parser.add_argument(
    #     "--continuous_step",
    #     action='store_true',
    #     help="whether use continuous step"
    # )
    # can tuing but always have the same shape
    parser.add_argument(
        '--distance_weights', 
        nargs='*', 
        type=float, 
        default=[1.00, 1.00, 1.00, 1.00, 1.00, 1.00], 
        help='List of distance matrix values'
    )
    parser.add_argument(
        '--reward_weights', 
        nargs='*', 
        type=float, 
        default=[1, 0.3, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 
        help='reward weights used in parking reward'
    )
    parser.add_argument(
        "--goal",
        type=parse_list,
        default=[0, 0, 0, 0, 0, 0],
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
    # parser.add_argument(
    #     '--penalty_backward',
    #     type=float,
    #     default=2.0,
    #     help='penalty for backward'
    # )
    # parser.add_argument(
    #     '--penalty_switch',
    #     type=float,
    #     default=10.0,
    #     help='penalty for switch'
    # )
    parser.add_argument(
        '--constraint_coeff',
        type=float,
        default=5.0,
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
    