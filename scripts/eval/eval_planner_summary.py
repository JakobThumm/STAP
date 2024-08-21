"""This file creates a summary of the planner evaluation results.

Author: Jakob Thumm
Date: 23.04.2024
"""

import argparse
import os
import pathlib
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def list_subdirs(parent_path):
    """Lists all subdirectories in the given parent directory.

    Arguments:
    - parent_path: The parent directory path as a string.

    Returns:
    - A list of strings, where each string is the full path to a subdirectory.
    """
    return [
        os.path.join(parent_path, name)
        for name in os.listdir(parent_path)
        if os.path.isdir(os.path.join(parent_path, name))
    ]


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

                # Debug
                trial_id = int(trial_name.split("_")[-1])
                # if trial_id % 13 == 4 or trial_id % 13 == 6 or trial_id % 13 == 12:
                #     print(f"Skipping trial {trial_name}")
                #     continue

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
    if "p_success" in keys:
        values = np.array([result["p_success"] for result in trial_results])
        results["shooting_failed"] = np.sum(values == 0) / len(values)

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
        "p_success",
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
            elif key == "p_success":
                # We want to summarize if the the last action was successful or not.
                values.extend([result[key] for result in results])
            # elif key == "observed_preference_values":
            #     # Emulate a threshold prefenerence function instead of the normal cdf function.
            #     for result in results:
            #         for val in result[key]:
            #             if not math.isnan(val):
            #                 values.append(1.0 if val > 0.02275 else 0.0)
            else:
                # Just take all values available
                for result in results:
                    values.extend(result[key])
        values = np.array(values).flatten()
        if key == "p_success":
            summary["shooting_failed_percent_mean"] = np.sum(values == 0) / len(values)
            summary["shooting_failed_percent_std"] = 0.0
            summary["shooting_failed_percent_ci_025"] = 0.0
            summary["shooting_failed_percent_ci_975"] = 0.0
        summary[key + "_mean"] = np.nanmean(values)
        summary[key + "_std"] = np.nanstd(values)
        ci = bootstrap(data=(values,), statistic=np.nanmean, axis=0, confidence_level=0.95).confidence_interval
        summary[key + "_ci_025"] = ci.low
        summary[key + "_ci_975"] = ci.high
    return summary


def dicts_to_csv(
    dict_list: Sequence[Dict[str, Any]],
    csv_filename: str,
    keys: Sequence[str] = [
        "values",
        "rewards",
        "predicted_preference_values",
        "observed_preference_values",
        "t_planner",
    ],
) -> None:
    """
    Converts a list of dictionaries into a pandas DataFrame and saves it as a CSV file.

    Arguments:
    - dict_list: A list of dictionaries, where each dictionary contains method results.
      Each dictionary should have the method name under a specific key if needed.
    - csv_filename: The filename for the output CSV.

    The function assumes all dictionaries have the same structure.
    """
    n_results = len(keys)
    n_experiments = len(dict_list)
    rows_list = []
    for i in range(n_results):
        result_dict = {"ID": i, "Measurement": keys[i]}
        for j in range(n_experiments):
            result_dict[f"Method{dict_list[j]['experiment']}Mean"] = dict_list[j][keys[i] + "_mean"]
            result_dict[f"Method{dict_list[j]['experiment']}CI025"] = dict_list[j][keys[i] + "_ci_025"]
            result_dict[f"Method{dict_list[j]['experiment']}CI975"] = dict_list[j][keys[i] + "_ci_975"]
        rows_list.append(result_dict)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows_list)

    # Save the DataFrame as a CSV file
    df.to_csv(csv_filename, index=False, sep=",")


def create_result_summary(
    eval_path: Union[str, pathlib.Path],
) -> None:
    experiment_paths = list_subdirs(eval_path)
    summary_list = []
    keys = [
        "p_success",
        "values",
        "rewards",
        "predicted_preference_values",
        "observed_preference_values",
        "t_planner",
    ]
    for experiment_path in experiment_paths:
        print(f"Summarizing results for experiment {experiment_path}")
        raw_data = read_in_files(experiment_path)
        summary = simplified_summary_all_trials(raw_data, keys)
        summary["experiment"] = experiment_path.split("/")[-1]
        summary_list.append(summary)
    keys.append("shooting_failed_percent")
    dicts_to_csv(summary_list, f"{eval_path}/summary.csv", keys)
    # for trial_name, trial_results in raw_data.items():
    #     summary = summarize_trial(trial_results)
    #     print(f"Summary for trial {trial_name}: {summary}")


def create_result_summary_by_trial(
    eval_path: Union[str, pathlib.Path],
) -> None:
    n_experiments = 10
    result_rows = [
        {"ID": i, "oracle_0": -1, "ablation_0": -1, "generated_0": -1, "generated_1": -1, "generated_2": -1}
        for i in range(n_experiments)
    ]
    keys = [
        "observed_preference_values",
    ]
    raw_data = read_in_files(eval_path)
    for trial_name, trial_results in raw_data.items():
        trial_id = int(trial_name.split("_")[-1])
        trial_type = trial_name.split("_")[0]
        trial_number = trial_id % n_experiments
        trial_version = trial_id // n_experiments
        summary = summarize_trial(trial_results, keys)
        result_rows[trial_number][f"{trial_type}_{trial_version}"] = summary[keys[0] + "_mean"]

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(result_rows)

    # Save the DataFrame as a CSV file
    df.to_csv(f"{eval_path}/summary_observed_preference_values.csv", index=False, sep=",")


def main(args: argparse.Namespace) -> None:
    create_result_summary(**vars(args))
    create_result_summary_by_trial(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-path", help="Path to evaluation results", default="models/eval/planning/object_arrangement/"
    )
    args = parser.parse_args()

    main(args)
