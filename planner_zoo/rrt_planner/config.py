import argparse
import numpy as np

def create_parser():
    # currently not in use
    parser = argparse.ArgumentParser(description="Configuration for planning TT using rrt")
    
    
    ## --------------------------------- basic setting --------------------------------------------------- ##
    
    parser.add_argument("--xi_max", type=float, default=(np.pi) / 4, help="Maximum steering angle")
    parser.add_argument("--goal_yaw_error", type=float, default=np.deg2rad(3.0), help="[rad] allowable yaw error for goal")
    parser.add_argument("--max_steer", type=float, default=0.6, help="[rad] maximum steering angle")
    parser.add_argument("--v_max", type=float, default=2.0, help="Maximum velocity")
    ## --------------------------------- expand setting ---------------------------------------------------- ##
    parser.add_argument("--move_step", type=float, default=0.2, help="[m] path interporate resolution")
    parser.add_argument("--n_steer", type=float, default=20.0, help="number of steer command")
    
    ## --------------------------------- collision_check setting ------------------------------------------- ##
    parser.add_argument("--collision_check_step", type=int, default=10, help="skip number for collision check")
    parser.add_argument("--extend_area", type=float, default=0.0, help="[m] map extend length")
    parser.add_argument(
        "--safe_d",
        type=float,
        default=0.0,
        help="the safe distance from the vehicle to obstacle"
    )
    ## ----------------------------------- model setting ------------------------------------------------ ##
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

    
    
    ## -------------------------------- planner setting ---------------------------------------------- ##
    parser.add_argument("--env_name",
                        type=str,
                        default="three_tractor_trailer",
                        choices=["single_tractor", "three_tractor_trailer"],
                        help="which env you choose to run")
    parser.add_argument("--obs",
                        type=bool,
                        default=True,
                        help="whether there is obstacle")
    parser.add_argument("--heuristic_type",
                        type=str,
                        default="traditional",
                        choices=["traditional", "critic"],
                        help="which heuristic you choose to use in hybrid astar")
    parser.add_argument("--heuristic_reso",
                        type=float,
                        default=0.5,
                        help="heuristic resolution [m]")
    parser.add_argument("--heuristic_rr",
                        type=float,
                        default=0.25,
                        help="heuristic rr [m]")
    parser.add_argument("--qp_type",
                        type=str,
                        default="heapq",
                        choices=["heapdict", "heapq"],
                        help="which qp to use")
    parser.add_argument(
        "--max_iter", 
        type=int,
        default=100000,
        help="maximum iteration of the main loop"
    )
    parser.add_argument(
        "--sample_rate", 
        type=float,
        default=0.4,
        help="sample rate for random selection"
    )
    
    return parser