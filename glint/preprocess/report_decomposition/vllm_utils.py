import json

import torch


def initialize_llm(llm_config):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    model_config = llm_config["model_config"]
    model_name = model_config["model"]

    dtype = model_config.get("dtype", "torch.bfloat16")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."))

    llm = LLM(
        model=model_name,
        tensor_parallel_size=model_config.get("tensor_parallel_size", 4),
        dtype=dtype,
        gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.95),
        enforce_eager=model_config.get("enforce_eager", False),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params_dict = llm_config["sampling_params"].copy()
    use_json_object = sampling_params_dict.pop("json_object", True)

    structured_outputs = None
    if use_json_object:
        structured_outputs = StructuredOutputsParams(json_object=True)

    sampling_params = SamplingParams(**sampling_params_dict, structured_outputs=structured_outputs)

    batch_size = llm_config.get("batch_size", 1000)
    max_count = llm_config.get("max_count", 5)

    return {
        "llm": llm,
        "tokenizer": tokenizer,
        "sampling_params": sampling_params,
        "batch_size": batch_size,
        "max_count": max_count,
    }


def extract_json_from_text(text):
    """Extract JSON from text, handling cases like "json{...}" or markdown code blocks."""
    text = text.replace("```", "").strip()

    text = text.strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()

    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_text = text[start_idx : end_idx + 1]
        return json_text

    return text


def parse_json_safely(text):
    try:
        json_object = json.loads(extract_json_from_text(text))
        return json_object
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from text: {text}")


def generate_batch_response(prompts, batch, llm_instance):
    llm = llm_instance.get("llm")
    tokenizer = llm_instance.get("tokenizer")
    sampling_params = llm_instance.get("sampling_params")
    max_count = llm_instance.get("max_count")

    prompts = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in prompts]

    outputs = llm.generate(prompts, sampling_params)

    text_outputs = []
    for i, data in enumerate(batch):
        text_outputs.append(outputs[i].outputs[0].text)

    success_json_list = []
    failed_json_list = []
    for i, data in enumerate(batch):
        try:
            generated_text = text_outputs[i].split("assistantfinal", 1)[1].strip()
        except IndexError:
            print(f"No 'assistantfinal' found in output[{i}]")
            generated_text = text_outputs[i].strip()

        attempts = 0
        result = None

        while attempts < max_count:
            try:
                result = parse_json_safely(generated_text)

                for k, v in result.items():
                    data[k] = v
                break

            except ValueError:
                if not data.get("raw_report", "").strip():
                    print("empty inputs!")
                    break

                attempts += 1
                print(f"Regenerate output because it is not in JSON format! : {attempts}")
                outputs = llm.generate([prompts[i]], sampling_params)
                generated_text = outputs[0].outputs[0].text

                try:
                    generated_text = generated_text.split("assistantfinal", 1)[1].strip()
                except IndexError:
                    generated_text = generated_text.strip()

        if attempts >= max_count:
            failed_json_list.append(data)
            result = None

        if result:
            success_json_list.append(data)

    return success_json_list, failed_json_list
