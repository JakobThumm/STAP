"""
Generate problems, solutions and corresponding natural language instructions for a given PDDL domain.

For now, the PDDL goal is a conjunction of propositions that are all true in the initial state.

The goal is generated by randomly sampling a number of objects and then randomly
placing the newest object on top of another object.

Usage:

PYTHONPATH=. python scripts/data/generate_lm_dataset.py --config.overwrite
"""

import random
from typing import List, Union

from symbolic import _and, parse_proposition
import symbolic
import tyro
import numpy as np

from typing import List
from configs.base_config import ProblemGenerationConfig
from temporal_policies.task_planners.lm_data_structures import InContextExample

from temporal_policies.task_planners.lm_utils import (
    predicate_scheme_to_python_syntax,
)


def parse_pred_arg_string(string: str) -> List[Union[str, List[str]]]:
    """
    Given a string of the form "(predicate arg1 arg2 arg3 ...)", return a list of the predicate and a list of the arguments.

    Example input: "(on red_box table)" (from problem._initial_state)
    """
    # Remove the parentheses from the string
    string = string.strip("()")
    # Split the string into a list of words
    words = string.split()
    # The first word is the predicate
    predicate = words[0]
    # The rest of the words are the arguments
    arguments = words[1:]
    return [predicate], arguments

def generate_instruction_for_on_proposition(object_a: str, object_b: str) -> List[str]:
    prepositions = [
        "on top of",
        "on",
        "above",
        "over",
        "onto",
        "on top of",
        "on the top of",
    ]
    verbs = [
        "place",
        "put",
        "move",
        "stack",
        "position",
        "situate",
        "arrange",
        "set",
        "get",
        "perch",
    ]
    return f"{random.choice(verbs)} {object_a} {random.choice(prepositions)} {object_b}"


def generate_instruction_for_inhand_proposition(object_a):
    verbs = ["hang on to", "grab", "pick up", "hold", "take", "carry, in your hand,"]
    return f"{random.choice(verbs)} the {object_a}"


def add_prefix_and_suffix_to_instruction(instruction: str) -> str:
    prefixes = [
        "",
        "please ",
        "Can you ",
        "Hey, ",
        "can you please ",
        "could you ",
        "could you please ",
    ]
    suffixes = ["", " please?", " - thanks", ", thanks!", " now, thanks!"]
    return random.choice(prefixes) + instruction + random.choice(suffixes)


def extract_propositions_from_pddl_goal(pddl_goal: str) -> List[str]:
    """Given a goal proposition, extract a list of the individual propositions with python syntax.

    Example:
        (and (inhand red_box) (on cyan_box table) (on blue_box table) (on yellow_box table))
        becomes ["inhand(red_box)", "on(cyan_box, table)", "on(blue_box, table)", "on(yellow_box, table)"]
    """
    assert (
        "not" not in pddl_goal and "or " not in pddl_goal
    ), "not and or not supported yet"  # use 'or ' to distinguish with 'workspace '

    # remove outer parentheses
    assert (
        pddl_goal[0] == "(" and pddl_goal[-1] == ")"
    ), "assume pddl goal is of form (and ...)"
    pddl_goal = pddl_goal[1:-1]

    # remove the "and" from the beginning
    assert pddl_goal[:4] == "and ", "assume pddl goal is of form (and ...)"
    pddl_goal = pddl_goal[4:]

    # split on left parentheses
    individual_propositions = pddl_goal.split("(")
    # strip right parentheses
    individual_propositions = [prop.strip() for prop in individual_propositions]
    individual_propositions = [prop.strip(")") for prop in individual_propositions]

    # remove empty strings
    individual_propositions = [prop for prop in individual_propositions if prop != ""]

    # convert individual propositions to have the form "predicate(arg1, arg2, ...)"
    individual_propositions = [
        predicate_scheme_to_python_syntax(prop) for prop in individual_propositions
    ]
    return individual_propositions


def generate_overall_instruction_for_pddl_problem(pddl):
    """
    Note: we use the pddl object to extract the goal propositions
    instead of the problem object because the goal propositions
    from problem are not formatted as nicely

    Note: unclear how bad of an idea it is to ignore the initial
    when generating/interpreting the language instruction --- likely important
    in some cases ...

    One example: for the llm to output predicates (in the way we do it)
    for an instruction with the word "all", it needs to know the initial state.
    Furthermore, if we the language uses other descriptions of objects,
    like "the blood-colored block", then we need to know the initial state to know
    which object is being referred to.

    Hardcoded rules:
        - use the word "all" if there are multiple objects in consistently satisfying
          a proposition
    """
    individual_propositions = extract_propositions_from_pddl_goal(str(pddl.goal))
    all_on_rack = True
    all_on_table = True
    inhand_propositions = []

    for prop in individual_propositions:
        if "on" in prop:
            if "table" not in prop:
                all_on_table = False
            if "rack" not in prop:
                all_on_rack = False
        if "inhand" in prop:
            inhand_propositions.append(prop)
    assert (
        len(inhand_propositions) <= 1
    ), "only maximum of one inhand propositions supported"

    if all_on_table:
        instruction = generate_instruction_for_on_proposition(
            "all the boxes", "the table"
        )
    elif all_on_rack:
        instruction = generate_instruction_for_on_proposition(
            "all the boxes", "the rack"
        )
    else:
        props = []
        for prop in individual_propositions:
            if "on" in prop:
                object_a, object_b = parse_proposition(prop)[1]
                # don't perform actions on the rack because rack is not movable?
                if object_a == "rack":
                    continue
                props.append(
                    generate_instruction_for_on_proposition(
                        "the " + object_a, "the " + object_b
                    )
                )
        instruction = " and ".join(props)

    if len(inhand_propositions) > 0:
        inhand_object = inhand_propositions[0].split("(")[1].split(")")[0]
        instruction += " and " + generate_instruction_for_inhand_proposition(
            inhand_object
        )
    return add_prefix_and_suffix_to_instruction(instruction)

def get_task_plan(
    pddl: symbolic.Pddl,
    max_depth: int = 10,
    timeout: int = 10,
) -> List[str]:
    """Get a plan for the given pddl problem. 
    Assumes that the problem is solvable (and also within the timeout).

    Args:
        pddl: the pddl problem

    Returns:
        A list of actions to take to solve the problem.
    """
    planner = symbolic.Planner(pddl, pddl.initial_state)
    bfs = symbolic.BreadthFirstSearch(
        planner.root, max_depth=max_depth, timeout=timeout, verbose=False
    )
    # take first (shortest) plan returned
    plan = list(bfs)[0]
    return [str(node.action) for node in plan[1:]]

def create_problem(
    config: ProblemGenerationConfig,
    pddl: symbolic.Pddl,
    problem_name: str,
    objects_with_properties: List,
    num_objects: int,
) -> None:
    problem: symbolic.Problem = symbolic.Problem(problem_name, pddl.name)

    rack = objects_with_properties[0]
    assert problem.add_object(rack[0], rack[1]), "Failed to add rack"
    # add rack to table
    problem.add_initial_prop(f"on({rack[0]}, table)")
    num_objects += 1

    # randomly add other objects
    while len(problem._objects) < num_objects:
        # randomly sample an object using random.choice
        obj = random.choice(objects_with_properties[1:])
        success = problem.add_object(obj[0], obj[1])
        if not success:
            continue
        # manually add table
        curr_objects = list(problem._objects) + [pddl.objects[0]]  
        # on predicate: place newly created object on another object 
        # that's not itself or a box that already has a box on it
        potential_on_objs = []
        for o in curr_objects:
            if o.name == obj[0]:
                continue
            for prop in problem._initial_state:
                pred, args = parse_pred_arg_string(prop)
                if pred == "on" and args[1] == o.name and "box" in o.name:
                    continue
            potential_on_objs.append(o)

        on_obj = random.choice(potential_on_objs)
        problem.add_initial_prop(f"on({obj[0]}, {on_obj.name})")

    # create the problem pddl file and save to disk
    with open(config.pddl_cfg.get_problem_file(problem_name), "w") as f:
        f.write(str(problem))

    # create a new pddl object to forward propagate state actions
    pddl = symbolic.Pddl(
        config.pddl_cfg.pddl_domain_file, config.pddl_cfg.get_problem_file(problem_name)
    )

    state = pddl.initial_state
    steps = random.randint(config.min_steps, config.max_steps)
    for _ in range(steps):
        valid_actions = pddl.list_valid_actions(state)
        if len(valid_actions) == 0:
            print(f"No valid actions. State: {state}")
            import ipdb; ipdb.set_trace()
            break

        action = random.choice(valid_actions)
        state = pddl.next_state(state, action)

    valid_syntax_state = []
    for prop in state:
        parsed_prop = parse_proposition(prop)
        if parsed_prop[0] == "inworkspace":
            continue
        valid_syntax_prop = f"({parsed_prop[0]} {' '.join(parsed_prop[1])})"
        valid_syntax_state.append(valid_syntax_prop)
    problem.set_goal(_and(*valid_syntax_state))

    # re-write the problem pddl file with the new goal
    with open(config.pddl_cfg.get_problem_file(problem_name), "w") as f:
        f.write(str(problem))

    pddl = symbolic.Pddl(
        config.pddl_cfg.pddl_domain_file,  config.pddl_cfg.pddl_problem_file
    )

    human_instruction = generate_overall_instruction_for_pddl_problem(pddl)

    task_plan = get_task_plan(pddl, max_depth=10, timeout=10)  # assumes this succeeds?

    example = InContextExample(
        predicates=["on(a, b)", "inhand(a)"],
        primitives=["pick(a, b)", "place(a, b)", "pull(a, b)", "push(a, b)"],
        scene_objects=[obj.name for obj in pddl.objects],
        scene_object_relationships=list(pddl.initial_state),
        human=human_instruction,
        goal=extract_propositions_from_pddl_goal(str(pddl.goal)),
        robot=task_plan,
        pddl_domain_file=config.pddl_cfg.pddl_domain_file,
        pddl_problem_file=config.pddl_cfg.pddl_problem_file,
    )
    
    example.save_to_json(
        config.pddl_cfg.get_prompt_file(problem_name), overwrite=config.overwrite
    )


def main(config: ProblemGenerationConfig):
    pddl = symbolic.Pddl(config.pddl_cfg.pddl_domain_file)
    if config.pddl_cfg.pddl_domain == "constrained_packing":
        objects_with_properties: List[str] = [
            ("rack", "unmovable"),
            ("red_box", "box"),
            ("yellow_box", "box"),
            ("cyan_box", "box"),
            ("blue_box", "box"),
        ]
    elif config.pddl_cfg.pddl_domain == "hook_reach":
        raise NotImplementedError("hook_reach not implemented properly yet (especially for the hooking related tasks)")
        objects_with_properties: List[str] = [
            ("rack", "unmovable"),
            ("red_box", "box"),
            ("yellow_box", "box"),
            ("cyan_box", "box"),
            ("blue_box", "box"),
            ("hook", "hook"),  # TODO(klin): unclear how the property field works
        ]
    else:
        raise NotImplementedError(f"{config.pddl_cfg.pddl_domain} not implemented yet")

    for problem_idx in range(config.num_problems):
        problem_name = f"{config.pddl_cfg.pddl_problem_prefix}{problem_idx}"

        num_objects = np.random.randint(2, len(objects_with_properties))
        create_problem(config, pddl, problem_name, objects_with_properties, num_objects)


if __name__ == "__main__":
    tyro.cli(main)
