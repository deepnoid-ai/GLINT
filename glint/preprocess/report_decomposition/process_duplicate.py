import os
from collections import defaultdict

from tqdm import tqdm
from utils import load_json, save_json


def parse_input_config(input_config):
    target_keys = input_config.get("target_keys")
    if not target_keys:
        raise ValueError("target_keys is required in input_config")

    parsed_config = {
        "target_keys": target_keys,
        "combination": input_config.get("combination", "concat"),
        "separator": input_config.get("separator", " "),
    }
    print(f"Parsed input config: {parsed_config}")
    return parsed_config


def get_raw_report(data, parsed_config):
    target_keys = parsed_config["target_keys"]
    combination = parsed_config["combination"]
    separator = parsed_config["separator"]

    missing_keys = [key for key in target_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Missing keys in data: {missing_keys}. Available keys in data: {list(data.keys())}")

    if combination == "concat":
        return separator.join([data[key] for key in target_keys])
    else:
        raise ValueError(f"Invalid combination: {combination}")


def deduplicate_reports(original_input_path, save_dir, input_config):
    """Group identical raw reports; write report_mapping.json and unique_raw_reports.json under save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    if input_config is None:
        raise ValueError("input_config is required for report processing")

    parse_config = parse_input_config(input_config)

    input_list = load_json(original_input_path)

    raw_report_to_indices = defaultdict(list)
    for idx, data in enumerate(input_list):
        raw_report = get_raw_report(data, parse_config)
        raw_report_to_indices[raw_report].append(idx)

    unique_raw_reports = list(raw_report_to_indices.keys())
    print(f"Total samples: {len(input_list)}, Unique reports: {len(unique_raw_reports)}")

    mapping_save_path = os.path.join(save_dir, "report_mapping.json")
    raw_reports_save_path = os.path.join(save_dir, "unique_raw_reports.json")

    mapping_info = {}
    raw_reports_data = []

    for unique_idx, raw_report in enumerate(unique_raw_reports):
        indices = raw_report_to_indices[raw_report]
        mapping_info[str(unique_idx)] = indices
        raw_reports_data.append(
            {
                "raw_report": raw_report,
                "mapping_indices": indices,
            }
        )

    mapping_data = {
        "total_samples": len(input_list),
        "unique_reports": len(unique_raw_reports),
        "mapping": mapping_info,  # unique_idx => indices list
    }

    save_json(mapping_data, mapping_save_path)
    print(f"Saved mapping info to: {mapping_save_path}")

    save_json(raw_reports_data, raw_reports_save_path)
    print(f"Saved raw reports to: {raw_reports_save_path}")

    return raw_reports_data, mapping_save_path, raw_reports_save_path


def map_results_back(
    unique_results,
    original_input_path,
    remove_original_keys=[],
    save_path=None,
    check_none=True,
):
    """Expand unique results back onto the original records via mapping_indices; save to save_path if given."""
    original_input_list = load_json(original_input_path)

    all_results = [None] * len(original_input_list)

    for unique_idx in tqdm(range(len(unique_results)), desc="Mapping results back to original input"):
        processed_result = unique_results[unique_idx]
        mapping_indices = processed_result.pop("mapping_indices")

        if "raw_report" in processed_result:
            processed_result.pop("raw_report")

        for idx in mapping_indices:
            result_data = original_input_list[idx].copy()

            for k in remove_original_keys:
                if k in result_data:
                    del result_data[k]

            for k, v in processed_result.items():
                result_data[k] = v

            all_results[idx] = result_data

    if check_none:
        assert None not in all_results, "Some indices were not mapped"
    else:
        print(f"Found {sum(1 for r in all_results if r is None)} None values in results")

    if save_path:
        save_json(all_results, save_path)
        print(f"Saved mapped results to: {save_path}")

    return all_results
