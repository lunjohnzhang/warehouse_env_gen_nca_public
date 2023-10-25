import os
import gc
import json
import time
import fire
import numpy as np
import docplex

import docplex.mp.conflict_refiner as cr

from docplex.mp.model import Context, Model
from env_search.utils import (manufacture_obj_types, manufacture_env_str2number,
                              manufacture_env_number2str, flip_one_e_to_s,
                              flip_tiles, read_in_manufacture_map,
                              format_env_str)
from env_search import MAP_DIR

from pprint import pprint


# Adds constraints that ensure exactly one object is present in each cell
#
# mdl:                the milp model
# all_objects:        a list of all object variables [[W_i], [P_i], ...]
def add_object_placement(mdl, all_objects):
    # Transpose the given matrix and ensure exactly one object per graph node
    for cur_node in zip(*all_objects):
        mdl.add_constraint(sum(cur_node) == 1)


# Adds reachability constraints to milp
#
# mdl:                the milp model
# graph:              an adjacency list
# source_objects:     objects that must reach the sink objects [[P_i], ...]
# sink_objects:       objects that must be reached by the source objects [[K_i], [D_i], ...]
# blocking_objects:   a list of object types that impede movement [[W_i], ...]
# cnt:                integer to remember number of times add_reachability function is called
#                     needed because different calls should produce params with different names
#
# post condition: these constraints ensure that a path exists from some source
#                 object to all sink objects
def add_reachability(mdl, graph, source_objects, sink_objects, blocking_objects,
                     cnt):
    # Transpose the blocking objects matrix so all blocking objects for
    # a given node are easily accessible.
    blocking = list(zip(*blocking_objects))

    # Setup a flow network for each edge in the graph
    n_nodes = len(graph)
    # Add a flow variable for each edge in the graph
    # flow: the flow leaving node i
    # rev: flow edges entering node i
    flow = [[] for i in range(n_nodes)]
    rev = [[] for i in range(n_nodes)]
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            f = mdl.continuous_var(name='p_{}_{}-{}'.format(i, j, cnt),
                                   lb=0,
                                   ub=n_nodes)
            flow[i].append(f)
            rev[j].append(f)

    # Add supply and demand variables for the source and sink
    supplies = []
    demands = []
    for i in range(n_nodes):
        f = mdl.continuous_var(name='p_s_{}-{}'.format(i, cnt),
                               lb=0,
                               ub=n_nodes)
        supplies.append(f)
        f = mdl.continuous_var(name='p_{}_t-{}'.format(i, cnt), lb=0, ub=1)
        demands.append(f)
    # Add a flow conservation constraint for each node (outflow == inflow)
    for i in range(n_nodes):
        mdl.add_constraint(supplies[i] + sum(rev[i]) == demands[i] +
                           sum(flow[i]))

    # Add capacity constraints for each edge ensuring that no flow passes through a blocking object
    for i, neighbors in enumerate(flow):
        blocking_limits = [n_nodes * b for b in blocking[i]]
        for f in neighbors:
            mdl.add_constraint(f + sum(blocking_limits) <= n_nodes)

    # Place a demand at this object location if it contains a sink type object.
    sinks = list(zip(*sink_objects))
    for i in range(n_nodes):
        mdl.add_constraint(sum(sinks[i]) == demands[i])

    # Allow this node to have supply if it contains a source object
    sources = list(zip(*source_objects))
    for i in range(n_nodes):
        capacity = sum(n_nodes * x for x in sources[i])
        mdl.add_constraint(supplies[i] <= capacity)


# Adds edit distance cost function and constraints for fixing the level with minimal edits.
#
# graph:              an adjacency list denoting allowed movement
# objects:            a list [([(T_i, O_i)], Cm, Cc), ...] representing the
#                     cost of moving each object by one edge (Cm) and the cost
#                     of an add or delete (Cc).
#                     T_i represents the object variable at node i.
#                     O_i is a boolean value denoting whether node i originally
#                     contained T_i.
# add_movement:       When True, edit distance objective will be used.
#                     Otherwise, hamming distance objective will be used.
def add_edit_distance(mdl, graph, objects, add_movement=True):
    costs = []
    if not add_movement:
        for objects_in_graph, cost_move, cost_change in objects:
            for cur_var, did_contain in objects_in_graph:
                if did_contain:
                    costs.append(cost_change * (1 - cur_var))
                else:
                    costs.append(cost_change * cur_var)

    else:
        for obj_id, (objects_in_graph, cost_move,
                     cost_change) in enumerate(objects):

            # Setup a flow network for each edge in the graph
            n_nodes = len(graph)
            # Add a flow variable for each edge in the graph
            # flow: the flow leaving node i
            # rev: flow edges entering node i
            flow = [[] for i in range(n_nodes)]
            rev = [[] for i in range(n_nodes)]
            for i, neighbors in enumerate(graph):
                for j in neighbors:
                    f = mdl.continuous_var(name='edit({})_{}_{}'.format(
                        obj_id, i, j),
                                           lb=0,
                                           ub=n_nodes)
                    costs.append(cost_move * f)
                    flow[i].append(f)
                    rev[j].append(f)

            # Add a supply if the object was in the current location.
            # Demands go everywhere.
            demands = []
            waste = []
            num_supply = 0
            for i, (cur_var, did_contain) in enumerate(objects_in_graph):
                f = mdl.continuous_var(name='edit({})_{}_t'.format(obj_id, i),
                                       lb=0,
                                       ub=1)
                demands.append(f)

                # Add a second sink that eats any flow that doesn't find a home.
                # The cost of this flow is deleting the object.
                f = mdl.continuous_var(name='edit({})_{}_t2'.format(obj_id, i),
                                       lb=0,
                                       ub=n_nodes)
                costs.append(cost_change * f)
                waste.append(f)

                # Flow conservation constraint (inflow == outflow)
                if did_contain:
                    # If we had a piece of this type in the current node, match it to the outflow
                    mdl.add_constraint(1 + sum(rev[i]) == demands[i] +
                                       sum(flow[i]) + waste[i])
                    num_supply += 1
                else:
                    mdl.add_constraint(
                        sum(rev[i]) == demands[i] + sum(flow[i]) + waste[i])

            # Ensure we place a piece of this type to match it to the demand.
            for (cur_var,
                 did_contain), node_demand in zip(objects_in_graph, demands):
                mdl.add_constraint(node_demand <= cur_var)

            # Ensure that the source and sink have the same flow.
            mdl.add_constraint(num_supply == sum(demands) + sum(waste))

    mdl.minimize(mdl.sum(costs))


def add_reachability_helper(source_labels, sink_labels, blocking_labels, mdl,
                            adj, objs, cnt):
    source_objects = [
        objs[manufacture_obj_types.index(label)] for label in source_labels
    ]
    sink_objects = [
        objs[manufacture_obj_types.index(label)] for label in sink_labels
    ]
    blocking_objects = [
        objs[manufacture_obj_types.index(label)] for label in blocking_labels
    ]
    add_reachability(mdl, adj, source_objects, sink_objects, blocking_objects,
                     cnt)


# Add constraints such that obj_t1 is next to obj_t2 and
# vice versa
#
# mdl:                the milp model
# graph:              an adjacency list
# objs:               all objects
# obj_t1:             object type 1 e.g. '@'
# obj_t2:             object type 2 e.g. 'e'
# def add_neighboaring_constraint(mdl, graph, objs, obj_t1, obj_t2):
#     for i, neighbors in enumerate(graph):
#         adj_var_t1 = []
#         adj_var_t2 = []
#         for j in neighbors:
#             adj_var_t1.append(objs[manufacture_obj_types.index(obj_t1)][j])
#             adj_var_t2.append(objs[manufacture_obj_types.index(obj_t2)][j])

#         # 1. Sum(variables of adj t2) >= variable of curr t1
#         # Make sure that if adj have no t2, curr node cannot be t1
#         # mdl.add_constraint(
#         #     sum(adj_var_t2) / 2 >= objs[manufacture_obj_types.index(obj_t1)][i])
#         mdl.add_constraint(
#             sum(adj_var_t2) >= objs[manufacture_obj_types.index(obj_t1)][i])

#         # 2. Sum(variables of adj t1) >= variable of curr t2
#         # Make sure that if adj have no t1, curr node cannot be t2
#         mdl.add_constraint(
#             sum(adj_var_t1) >= objs[manufacture_obj_types.index(obj_t2)][i])


# Add constraints such that one instance of obj_t1 is next to one instance of
# obj_t2 and vice versa
#
# mdl:                the milp model
# graph:              an adjacency list
# objs:               all objects
# objs_t1:            list of object type 1 e.g. ['0', '1', '2']
# objs_t2:            list of object type 2 e.g. ['e']
def add_neighboaring_constraint(mdl, graph, objs, objs_t1, objs_t2):
    objs_non_t2 = []
    for obj in manufacture_obj_types:
        if obj not in objs_t2:
            objs_non_t2.append(obj)
    for i, neighbors in enumerate(graph):
        adj_var_t1 = []
        adj_var_t2 = []
        adj_var_non_t2 = []
        for j in neighbors:
            for obj_t1 in objs_t1:
                adj_var_t1.append(objs[manufacture_obj_types.index(obj_t1)][j])
            for obj_t2 in objs_t2:
                adj_var_t2.append(objs[manufacture_obj_types.index(obj_t2)][j])
            for obj_non_t2 in objs_non_t2:
                adj_var_non_t2.append(objs[manufacture_obj_types.index(obj_non_t2)][j])

        var_t1 = []
        var_t2 = []
        for obj_t1 in objs_t1:
            var_t1.append(objs[manufacture_obj_types.index(obj_t1)][i])
        for obj_t2 in objs_t2:
            var_t2.append(objs[manufacture_obj_types.index(obj_t2)][i])

        # 1. Sum(variables of adj t2) >= variable of curr t1
        # Make sure that if adj have no t2, curr node cannot be t1

        # There is at least one endpoint around each obstacle.
        mdl.add_constraint(sum(adj_var_t2) >= sum(var_t1))

        # 2. Sum(variables of adj t1) >= variable of curr t2
        # Make sure that if adj have no t1, curr node cannot be t2
        mdl.add_constraint(sum(adj_var_t1) >= sum(var_t2))


# env_np: environment in numpy array format.
# agent_num: number of agents
# warm_env_np: list of warm up solution in numpy array format.
# time_limit: in seconds
# add_movement: if True, use edit distance obj, otherwise, use hamming distance.
# NOTE: For manufacture system repair, "shelf" refers to manufacture stations
def repair_env(
    env_np,
    warm_envs_np=None,
    time_limit=60,
    add_movement=True,
    max_n_shelf=25,
    min_n_shelf=10,
    seed=0,
    limit_n_shelf=True,
    n_threads=1,
    agent_num=60,
):
    # # Flip one of e to s in warm up solutions
    # env_np = flip_one_r_to_s(env_np)
    if warm_envs_np is not None:
        for i, warm_env_np in enumerate(warm_envs_np):
            warm_envs_np[i] = flip_one_e_to_s(warm_env_np,
                                              obj_types=manufacture_obj_types)

    n, m = env_np.shape

    deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # Build an adjacency list for the dynamics of Manufacture
    n_nodes = n * m
    adj = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        cur_row = i // m
        cur_col = i % m
        for dr, dc in deltas:
            nxt_row = cur_row + dr
            nxt_col = cur_col + dc
            if 0 <= nxt_row < n and 0 <= nxt_col < m:
                j = nxt_row * m + nxt_col
                adj[i].append(j)

    context = Context.make_default_context()
    context.cplex_parameters.threads = n_threads
    context.cplex_parameters.dettimelimit = time_limit * 1000
    if seed is not None:
        context.cplex_parameters.randomseed = seed

    with Model(context=context) as mdl:

        # Set parameters
        mdl.parameters.mip.pool.relgap = 0.05

        objs = []
        for obj_label in manufacture_obj_types:
            curr_type = [
                mdl.integer_var(name='obj_{}_{}'.format(obj_label, i),
                                lb=0,
                                ub=1) for i in range(n_nodes)
            ]
            objs.append(curr_type)

        # ensure one cell contains one obj_type
        add_object_placement(mdl, objs)

        # Specify Number of shelves
        # We want the sum of all categories of shelves to be limited
        if limit_n_shelf:
            mdl.add_constraint(
                sum(objs[manufacture_obj_types.index("0")]) +
                sum(objs[manufacture_obj_types.index("1")]) +
                sum(objs[manufacture_obj_types.index("2")]) >= min_n_shelf)
            mdl.add_constraint(
                sum(objs[manufacture_obj_types.index("0")]) +
                sum(objs[manufacture_obj_types.index("1")]) +
                sum(objs[manufacture_obj_types.index("2")]) <= max_n_shelf)

        # We want at least one instance of 0, 1, 2 shelf
        mdl.add_constraint(sum(objs[manufacture_obj_types.index("0")]) >= 1)
        mdl.add_constraint(sum(objs[manufacture_obj_types.index("1")]) >= 1)
        mdl.add_constraint(sum(objs[manufacture_obj_types.index("2")]) >= 1)

        # We need enough empty space for the agents
        mdl.add_constraint(
            n * m - (sum(objs[manufacture_obj_types.index("0")]) +
                     sum(objs[manufacture_obj_types.index("1")]) +
                     sum(objs[manufacture_obj_types.index("2")])) >= agent_num)

        # Only one 's'
        # NOTE: 's' in manufacture system is different from that in kiva.
        # In kiva, 's' is flipped from one of 'r' and is fixed.
        # In manufacture, 's' is essentially a special 'e' and can be anywhere.
        # We need 's' in both systems to encode the reachability constraints.
        # We will flip it back to 'e' at the end.
        mdl.add_constraint(sum(objs[manufacture_obj_types.index("s")]) == 1)

        # Obstacle must be next to at least one endpoint and vice versa
        add_neighboaring_constraint(mdl, adj, objs, ["0", "1", "2"], ["e", "s"])

        # reachability
        source_labels = "s"
        sink_labels = "e."
        blocking_labels = "012"
        add_reachability_helper(source_labels, sink_labels, blocking_labels,
                                mdl, adj, objs, 0)

        # add edit distance objective
        objects = []
        cost_move = 1
        cost_change = 20
        for cur_idx, cur_obj in enumerate(objs):
            objects_in_graph = []
            for r in range(n):
                for c in range(m):
                    i = r * m + c
                    objects_in_graph.append((cur_obj[i], cur_idx == env_np[r,
                                                                           c]))
            objects.append((objects_in_graph, cost_move, cost_change))

        add_edit_distance(mdl, adj, objects, add_movement=add_movement)

        # Add warm up solution to the model, if any.
        if warm_envs_np is not None:
            for warm_env_np in warm_envs_np:
                warm_env_str = manufacture_env_number2str(warm_env_np)
                warm_sol = mdl.new_solution()
                for i, row in enumerate(warm_env_str):
                    for j, tile in enumerate(row):
                        id = i * m + j
                        warm_sol.add_var_value('obj_{}_{}'.format(tile, id), 1)
                        for c in manufacture_obj_types[:-1]:
                            if c != tile:
                                warm_sol.add_var_value(
                                    'obj_{}_{}'.format(c, id), 0)
                mdl.add_mip_start(
                    warm_sol,
                    effort_level=docplex.mp.constants.EffortLevel.SolveFixed)

        solution = mdl.solve()

        if solution is None:
            print("No solution")
            return None

        def get_idx_from_variables(solution, node_id):
            for i, obj_var in enumerate(objs):
                if round(solution.get_value(obj_var[node_id])) == 1:
                    return i
            return -1

        # Extract the new level from the milp model
        new_env = np.zeros((n, m))
        for r in range(n):
            for c in range(m):
                i = r * m + c
                new_env[r, c] = get_idx_from_variables(solution, i)

        del solution
        gc.collect()

        new_env = new_env.astype(np.uint8)

        # Flip s back to e
        new_env = flip_tiles(new_env, 's', 'e', obj_types=manufacture_obj_types)
        if warm_envs_np is not None:
            for i, warm_env_np in enumerate(warm_envs_np):
                warm_envs_np[i] = flip_tiles(
                    warm_env_np,
                    's',
                    'e',
                    obj_types=manufacture_obj_types,
                )

        # print(f"Repair Obj value: {mdl.objective_value}")
        assert env_np.shape == new_env.shape
        return new_env


def main(
    map_filepath,
    warm_filepath=None,
    add_movement=True,
    n_threads=1,
):
    # Read in envs
    env_str, _ = read_in_manufacture_map(map_filepath)
    env_np = manufacture_env_str2number(env_str)
    env_np = env_np.astype(np.uint8)

    warm_env_np = None
    if warm_filepath is not None:
        warm_env_str, _ = read_in_manufacture_map(warm_filepath)
        warm_env_np = manufacture_env_str2number(warm_env_str)
        warm_env_np = warm_env_np.astype(np.uint8)

    # Fix env
    before_fix = format_env_str(manufacture_env_number2str(env_np))
    print("Before repair:")
    print(before_fix)
    repair_start_time = time.time()
    repaired_env = repair_env(
        env_np,
        warm_envs_np=[warm_env_np] if warm_env_np is not None else None,
        add_movement=add_movement,
        max_n_shelf=1188,
        min_n_shelf=0,
        n_threads=n_threads,
        agent_num=60,
    )
    after_fix = format_env_str(
        manufacture_env_number2str(repaired_env.astype(np.uint8)))
    print("\nAfter repair:")
    print(after_fix)
    time_spent = str(round(time.time() - repair_start_time, 2))
    print(f"Spend {time_spent} seconds on repairing.")


if __name__ == "__main__":
    fire.Fire(main)
