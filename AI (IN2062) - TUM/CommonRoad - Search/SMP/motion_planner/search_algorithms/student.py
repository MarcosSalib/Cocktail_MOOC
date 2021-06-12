import numpy as np
from abc import abstractmethod, ABC

from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.trajectory import State
from commonroad.scenario.trajectory import Trajectory

from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import BestFirstSearch
from SMP.motion_planner.search_algorithms import base_class
# from SMP.motion_planner.search_algorithms import helpers

class StudentMotionPlanner(BestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)


    def evaluation_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # todo: Implement your own evaluation function here.                   #
        ########################################################################
        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)

    def heuristic_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # todo: Implement your own heuristic cost calculation here.            #
        # Hint:                                                                #
        #   Use the State of the current node and the information from the     #
        #   planning problem, as well as from the scenario.                    #
        #   Some helper functions for your convenience can be found in         #
        #   ./search_algorithms/base_class.py                             #
        ########################################################################

        h1 = self.heuristic_1(node_current)
        h2 = self.heuristic_2(node_current)
        h3 = self.heuristic_3(node_current)

        list_ = np.array([h1, h2, h3], dtype=np.float64)
        return np.max(list_)

    def heuristic_1(self, node_current: PriorityNode) -> float:
        """
        original time-based
        """
        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if 'time_step' in self.planningProblem.goal.state_list[0].attributes:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

    def heuristic_2(self, node_current: PriorityNode) -> float:

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if 'position' in self.planningProblem.goal.state_list[0].attributes:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step
        else:
            velocity = node_current.list_paths[-1][-1].velocity

            if np.isclose(velocity, 0):
                return np.inf
            else:
                return self.calc_path_efficiency(node_current.list_paths[-1])

    def heuristic_3(self, node_current: PriorityNode) -> float:

        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if np.isclose(node_current.list_paths[-1][-1].orientation, 0):
            return np.inf

        if 'orientation' in self.planningProblem.goal.state_list[0].attributes:
            return self.calc_angle_to_goal(node_current.list_paths[-1][-1])