"""This file creates a summary of the planner evaluation results.

Author: Jakob Thumm
Date: 23.04.2024
"""

import argparse
import os
import pathlib
from typing import Any, Dict, Sequence, Union

import numpy as np
from scipy.stats import bootstrap


def read_in_files(
    base_path: Union[str, pathlib.Path],
) -> Dict[str, Sequence[Dict[str, Any]]]:
    """Reads in all results files in a given folder.

    Args:
        base_path: Path to the folder containing the evaluation results.

    Returns:
        Dictionary containing the results of all trials.
        trial_results[trial_name] = [results_0, results_1, ...]
        Each results dictionary contains the data from a single run.
    """
    # generated_ablation_trial_0/policy_cem_arrangement_no_custom_fns/results_0.npz
    trial_results = {}

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith("results_") and file.endswith(".npz"):
                # Identify the trial name by parsing the directory path
                trial_name = os.path.basename(os.path.dirname(root))  # or root.split(os.sep)[-1] for deeper nesting

                # Load the npz file
                file_path = os.path.join(root, file)
                data = np.load(file_path)

                # Add the loaded data to the trial_results dictionary
                if trial_name not in trial_results:
                    trial_results[trial_name] = [data]
                else:
                    trial_results[trial_name].append(data)

    return trial_results


def summarize_trial(
    trial_results: Sequence[Dict[str, Any]],
    keys: Sequence[str] = [
        "p_success",
        "values",
        "rewards",
        "p_visited_success",
        "predicted_preference_values",
        "observed_preference_values",
        "t_planner",
    ],
) -> Dict[str, Any]:
    """Summarizes the results of a single trial.

    Args:
        trial_results: Sequence of dictionaries containing the results of a single trial.

    Returns:
        Dictionary containing the summarized results.
    """
    results = {}
    for key in keys:
        if key == "t_planner":
            values = np.array([np.sum(result[key]) for result in trial_results])
        else:
            values = np.array([result[key] for result in trial_results])
        results[key + "_mean"] = np.nanmean(values)
        results[key + "_std"] = np.nanstd(values)
        ci = bootstrap(
            data=(values.flatten(),), statistic=np.nanmean, axis=0, confidence_level=0.95
        ).confidence_interval
        results[key + "_ci_025"] = ci.low
        results[key + "_ci_975"] = ci.high
    return results


def simplified_summary_all_trials(
    trial_results: Dict[str, Sequence[Dict[str, Any]]],
    keys: Sequence[str] = [
        "values",
        "rewards",
        "predicted_preference_values",
        "observed_preference_values",
        "t_planner",
    ],
) -> Dict[str, Any]:
    """Summarizes the results of all trials.

    Args:
        trial_results: Dictionary containing the results of all trials.

    Returns:
        Dictionary containing the summarized results.
    """
    summary = {}
    for key in keys:
        values = []
        for results in trial_results.values():
            if key == "t_planner":
                # Sum up the entire planning time per run
                values.extend([sum(result[key]) for result in results])
            elif key == "rewards":
                # We want to summarize if the the last action was successful or not.
                values.extend([result[key][-1] for result in results])
            else:
                # Just take all values available
                for result in results:
                    values.extend(result[key])
        values = np.array(values).flatten()
        summary[key + "_mean"] = np.nanmean(values)
        summary[key + "_std"] = np.nanstd(values)
        ci = bootstrap(data=(values,), statistic=np.nanmean, axis=0, confidence_level=0.95).confidence_interval
        summary[key + "_ci_025"] = ci.low
        summary[key + "_ci_975"] = ci.high
    return summary


def create_result_summary(
    eval_path: Union[str, pathlib.Path],
) -> None:
    raw_data = read_in_files(eval_path)
    simplified_trial_runs = simplified_summary_all_trials(raw_data)
    # for trial_name, trial_results in raw_data.items():
    #     summary = summarize_trial(trial_results)
    #     print(f"Summary for trial {trial_name}: {summary}")


def main(args: argparse.Namespace) -> None:
    create_result_summary(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-path", help="Path to evaluation results", default="models/eval/planning/object_arrangement/"
    )
    args = parser.parse_args()

    main(args)
