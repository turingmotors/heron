import json
import os

import fire
import torch
import yaml
from PIL import Image
from tqdm import tqdm

import wandb
from heron.models.prepare_processors import get_processor
from heron.models.utils import load_model, load_pretrained_weight


def generate_response(question, image, model, processor, device):
    """
    Generates a response for a given question and image.
    """
    text = f"##human: {question}\n##gpt: "
    inputs = processor(text=text, images=image, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(device).half()

    eos_token_id_list = [
        processor.tokenizer.pad_token_id,
        processor.tokenizer.eos_token_id,
        int(processor.tokenizer.convert_tokens_to_ids("\n")),
    ]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=256,
            do_sample=False,
            temperature=0.0,
            eos_token_id=eos_token_id_list,
            no_repeat_ngram_size=2,
        )
    return processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def load_questions(path):
    """
    Loads questions from a JSONL file.
    """
    with open(path, "r") as file:
        return [json.loads(line) for line in file]


def process_questions(img_root, questions, model, processor, device, verbose):
    """
    Processes a list of questions, generating answers for each.
    """
    results = []
    for q in tqdm(questions):
        image = Image.open(os.path.join(img_root, f"COCO_val2014_{q['image']}"))
        question = q["text_JA"]
        answer = generate_response(question, image, model, processor, device)
        if verbose:
            print(
                f"### ID: {q['question_id']}\n## question: {q['text_JA']}\n## answer: {answer}\n"
            )
        q["answer"] = answer
        results.append(q)
    return results


def upload_results(img_root, results, name):
    """
    Uploads the results to Weights & Biases.
    """
    project_name = os.getenv("WANDB_PROJECT_NAME", "default-project")
    wandb.init(project=project_name, name=name)
    table = wandb.Table(columns=["ID", "Name", "Image", "Question", "Answer"])
    for r in results:
        image = wandb.Image(
            Image.open(os.path.join(img_root, f"COCO_val2014_{r['image']}")), caption=r["answer"]
        )
        table.add_data(r["question_id"], name, image, r["text_JA"], r["answer"])
    wandb.log({"Table": table})


def save_results(results, output_path, model_name):
    """
    Saves the results to a JSONL file.
    """
    with open(os.path.join(output_path, f"{model_name}_answers.jsonl"), "w") as file:
        for r in results:
            file.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(
    config_file: str,
    questions_path: str,
    img_root: str,
    output_path: str,
    device: int = 0,
    is_upload_result: bool = False,
    verbose: bool = False,
):
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)
        model_config = config["model_config"]

    # make output dir
    os.makedirs(output_path, exist_ok=True)

    # load model
    model = load_model(model_config).to(device)
    print("Model loaded")

    # load pretrained weight
    if model_config.get("pretrained_path") is not None:
        print("load pretrained")
        load_pretrained_weight(model, model_config["pretrained_path"])
        print(f'Successfully loading pretrained weights from {model_config["pretrained_path"]}')

    # get preprocessor
    processor = get_processor(model_config)
    print("Processor loaded")

    questions = load_questions(questions_path)

    print("Start inference")
    results = process_questions(img_root, questions, model, processor, device, verbose)
    print("Done inference")

    output_model_name = config_file.split("/")[-1].split(".yml")[0]
    print("Saving results...")
    save_results(results, output_path, output_model_name)
    if is_upload_result:
        print("Upload to wandb...")
        upload_results(img_root, results, output_model_name)
    print("Done all evaluation")


if __name__ == "__main__":
    fire.Fire(main)
