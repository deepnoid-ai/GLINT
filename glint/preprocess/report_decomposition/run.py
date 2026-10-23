import argparse
import os
import traceback

from process_duplicate import (
    deduplicate_reports,
    map_results_back,
)
from process_steps import (
    get_step_config,
    get_step_prompt_and_config,
    process_batch_with_prompt,
)
from tqdm import tqdm
from utils import (
    get_step_file_path,
    load_config,
    load_json,
    save_json,
    save_snapshot_prompt,
)
from vllm_utils import initialize_llm


def create_step_function(system_prompt, llm_instance, step_config):
    def step_function(batch):
        return process_batch_with_prompt(
            batch=batch,
            system_prompt=system_prompt,
            llm_instance=llm_instance,
            input_key=step_config["input_key"],
            output_key=step_config["output_key"],
        )

    return step_function


def process_step_with_batch(step_function, current_data, batch_size):
    all_success_data = []
    all_failed_data = []

    for batch_idx in tqdm(range(0, len(current_data), batch_size)):
        batch = current_data[batch_idx : batch_idx + batch_size]

        try:
            result = step_function(batch)

            success_batch, failed_batch = result
            all_success_data.extend(success_batch)
            if failed_batch:
                all_failed_data.extend(failed_batch)

        except Exception as e:
            print(f"Error in batch {batch_idx // batch_size + 1}/{(len(current_data) - 1) // batch_size + 1}: {e}")
            traceback.print_exc()
            all_failed_data.extend(batch)

    return all_success_data, all_failed_data


def execute_pipeline_step(
    step_name,
    step_idx,
    current_data,
    intermediate_save_dir,
    dataset_type,
    step_configs,
    llm_instance,
):
    print(f"\n[Step {step_idx}] Running step: {step_name}")

    step_stats = {
        "step_name": step_name,
        "input_count": len(current_data),
        "output_count": 0,
        "failed_count": 0,
    }

    try:
        system_prompt = get_step_prompt_and_config(step_name, dataset_type)
        step_config = get_step_config(step_name, step_configs)
        step_function = create_step_function(system_prompt, llm_instance, step_config)

        result = process_step_with_batch(
            step_function=step_function,
            current_data=current_data,
            batch_size=llm_instance["batch_size"],
        )

        current_data, failed_data = result

        if failed_data:
            failed_save_path = get_step_file_path(intermediate_save_dir, step_idx, step_name, "failed")
            save_json(failed_data, failed_save_path)
            print(f"Saved failed cases to: {failed_save_path}")
            step_stats["failed_count"] = len(failed_data)

        step_stats["output_count"] = len(current_data)

        after_save_path = get_step_file_path(intermediate_save_dir, step_idx, step_name, "after")
        save_json(current_data, after_save_path)
        print(f"Saved state after {step_name} to: {after_save_path}")
        print(f"  Input: {step_stats['input_count']} samples")
        print(f"  Output: {step_stats['output_count']} samples")
        if step_stats["failed_count"] > 0:
            print(f"  Failed: {step_stats['failed_count']} samples")

        return current_data, failed_data, step_stats, None

    except Exception as e:
        print(f"Error executing step '{step_name}': {e}")
        traceback.print_exc()
        step_stats["error"] = str(e)

        error_save_path = get_step_file_path(intermediate_save_dir, step_idx, step_name, "error_state")
        save_json(current_data, error_save_path)
        print(f"Saved error state to: {error_save_path}")

        return current_data, None, step_stats, str(e)


def process_input_file(
    original_file_path,
    save_dir,
    split,
    dataset_config,
    pipeline_config,
    llm_instance,
):
    """Run one split through the pipeline, saving every intermediate output."""
    print(f"Original file path: {original_file_path}")

    intermediate_save_dir = os.path.join(save_dir, "intermediate", split)
    os.makedirs(intermediate_save_dir, exist_ok=True)

    stats = {"split": split, "original_samples": 0, "unique_reports": 0, "steps": {}}

    print(f"\n[Step 0] Deduplicating reports from: {original_file_path}")
    deduplicated_save_dir = os.path.join(intermediate_save_dir, "00_deduplicate")
    input_config = dataset_config.get("input_config")
    if input_config is None:
        raise ValueError("input_config is required for report processing")

    deduplicated_reports, mapping_save_path, raw_reports_save_path = deduplicate_reports(
        original_file_path, deduplicated_save_dir, input_config
    )

    stats["original_samples"] = len(load_json(original_file_path))
    stats["unique_reports"] = len(deduplicated_reports)
    print(f"Saved initial unique reports and mapping info to: {deduplicated_save_dir}")

    current_data = deduplicated_reports.copy()
    all_failed_cases = {}

    pipeline_steps = pipeline_config.get("steps")
    step_configs = pipeline_config.get("step_configs", {})
    dataset_type = dataset_config.get("dataset_type")
    remove_original_keys = dataset_config.get("remove_original_keys", [])

    if not isinstance(pipeline_steps, list) or len(pipeline_steps) == 0:
        raise ValueError("pipeline_steps must be a non-empty list")

    for step_idx, step_name in enumerate(pipeline_steps, start=1):
        current_data, failed_data, step_stats, error = execute_pipeline_step(
            step_name=step_name,
            step_idx=step_idx,
            current_data=current_data,
            intermediate_save_dir=intermediate_save_dir,
            dataset_type=dataset_type,
            step_configs=step_configs,
            llm_instance=llm_instance,
        )

        stats["steps"][step_name] = step_stats

        if error:
            continue

        if failed_data:
            all_failed_cases[step_name] = failed_data

    if all_failed_cases:
        all_failed_save_path = os.path.join(intermediate_save_dir, "all_failed_cases.json")
        save_json(all_failed_cases, all_failed_save_path)
        print(f"\nSaved all failed cases summary to: {all_failed_save_path}")

    print("\n[Final Step] Mapping results back to original input...")
    final_save_path = os.path.join(save_dir, f"{split}.json")
    all_results = map_results_back(
        unique_results=current_data,
        original_input_path=original_file_path,
        remove_original_keys=remove_original_keys,
        save_path=final_save_path,
    )

    stats["final_samples"] = len(all_results)
    print(f"Successfully processed {len(all_results)} samples for split: {split}")

    stats_save_path = os.path.join(intermediate_save_dir, "processing_stats.json")
    save_json(stats, stats_save_path)
    print(f"Saved processing statistics to: {stats_save_path}")

    return all_results, stats


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--add_cfg_list",
        type=str,
        nargs="+",
        required=True,
        help="Config file name list(e.g., decomposition mimic). Will load common.yaml automatically.",
    )
    args = parser.parse_args()
    config = load_config(args.add_cfg_list)

    data_root = config["dataset"]["data_root"]
    if not data_root:
        raise SystemExit("Set DATA_ROOT to the dataset root directory.")
    original_dir = os.path.join(data_root, config["dataset"]["original_dir"])
    save_dir = os.path.join(data_root, config["dataset"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    save_snapshot_prompt(save_dir, config)

    llm_instance = initialize_llm(config["llm_config"])

    for split in config["dataset"]["splits"]:
        original_file_path = os.path.join(original_dir, f"{split}.json")
        if not os.path.exists(original_file_path):
            raise FileNotFoundError(f"Original file {original_file_path} not found")

        process_input_file(
            original_file_path,
            save_dir,
            split,
            dataset_config=config["dataset"],
            pipeline_config=config["pipeline"],
            llm_instance=llm_instance,
        )


if __name__ == "__main__":
    main()
