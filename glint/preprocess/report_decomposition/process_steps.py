import importlib
import json

from vllm_utils import generate_batch_response


def get_step_prompt_and_config(step_name, dataset_type):
    """Return the system prompt of prompts/<step_name>.py followed by its {dataset_type} example."""
    prompt_module_path = f"prompts.{step_name}"
    try:
        prompt_module = importlib.import_module(prompt_module_path)
    except ImportError as e:
        raise ImportError(f"Prompt module {prompt_module_path} not found: {e}")

    prompt_name = f"{step_name.upper()}_PROMPT"

    if not hasattr(prompt_module, prompt_name):
        raise AttributeError(f"Prompt {prompt_name} not found in {prompt_module_path}")

    system_prompt = getattr(prompt_module, prompt_name)

    dataset_type_upper = dataset_type.upper()
    example_name = f"{dataset_type_upper}_{step_name.upper()}_EXAMPLE"

    if not hasattr(prompt_module, example_name):
        raise AttributeError(f"Example {example_name} not found in {prompt_module_path}")

    example = getattr(prompt_module, example_name)

    full_prompt = system_prompt + "\n\n" + example

    return full_prompt


def get_step_config(step_name, step_configs):
    if step_name not in step_configs:
        raise ValueError(
            f"Step '{step_name}' not found in pipeline.step_configs. Available steps: {list(step_configs.keys())}"
        )

    step_config = step_configs[step_name]
    input_key = step_config.get("input_key")
    output_key = step_config.get("output_key")

    if input_key is None:
        raise ValueError(f"input_key not found for step '{step_name}' in config")
    if output_key is None:
        raise ValueError(f"output_key not found for step '{step_name}' in config")

    return {
        "input_key": input_key,
        "output_key": output_key,
    }


def format_input_for_prompt(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, (list, dict)):
        return json.dumps(value, indent=2)
    else:
        return str(value)


def process_batch_with_prompt(
    batch,
    system_prompt,
    llm_instance,
    input_key,
    output_key=None,
    input_format="INPUT:\n{input}\n\nOUTPUT:",
):
    prompts = []
    for data in batch:
        if input_key not in data:
            raise ValueError(f"Missing '{input_key}' in data. Available keys: {list(data.keys())}")

        input = data[input_key]

        user_content = input_format.format(input=format_input_for_prompt(input))

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompts.append(prompt)

    success_data, failed_data = generate_batch_response(
        prompts=prompts,
        batch=batch,
        llm_instance=llm_instance,
    )

    if output_key and success_data:
        validated_success = []
        validated_failed = []
        for data in success_data:
            if output_key in data:
                validated_success.append(data)
            else:
                validated_failed.append(data)

        if validated_failed:
            failed_data.extend(validated_failed)
        success_data = validated_success

    return success_data, failed_data
