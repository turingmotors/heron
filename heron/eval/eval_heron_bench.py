import argparse
import asyncio
import json
import os
from collections import defaultdict

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

API_URL = "https://api.openai.com/v1/chat/completions"
API_HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json",
}
API_PAYLOAD = {
    "model": "gpt-4-0125-preview",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the quality of the answer.",
        },
    ],
    "temperature": 0.0,
}
NUM_SECONDS_TO_SLEEP = 0.5
TOTAL_QUESTIONS = 103


async def get_eval(session, content: str, max_tokens: int):
    payload = API_PAYLOAD.copy()
    payload["messages"].append({"role": "user", "content": content})
    payload["max_tokens"] = max_tokens

    while True:
        async with session.post(API_URL, headers=API_HEADERS, json=payload) as resp:
            if resp.status == 429:
                await asyncio.sleep(NUM_SECONDS_TO_SLEEP)
                continue
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


def parse_score(review):
    try:
        score_pair = review.split("\n")[0].replace(",", " ")
        score1, score2 = map(float, score_pair.split())
        return [score1, score2]
    except (IndexError, ValueError):
        print(f"Error parsing score from review: {review}")
        return [-1, -1]


def load_jsonl(path, num):
    scores = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            d = json.loads(line)
            scores[d["category"]].append(d["tuple"][1])
            scores[d["category"] + "_ref"].append(d["tuple"][0])
    return scores


def load_model_results(model_results, num=90):
    results = {}
    for model_name, result_path in model_results.items():
        scores = load_jsonl(result_path, num)
        result = {
            c: 100 * np.mean(s) / np.mean(scores[c + "_ref"])
            for c, s in scores.items()
            if "ref" not in c
        }
        results[model_name] = result
    return results


def plot_result(model_results, save_plot_name, min_value=0, max_value=100):
    labels = list(next(iter(model_results.values())).keys())
    model_scores = {
        model_name: [max(0, result[label]) for label in labels] + [max(0, result[labels[0]])]
        for model_name, result in model_results.items()
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["r", "g", "b", "c", "m", "y", "k", "w"]

    for i, (model_name, score) in enumerate(model_scores.items()):
        color = colors[i % len(colors)]
        ax.plot(angles, score, color=color, linewidth=2, label=model_name)
        ax.fill(angles, score, color=color, alpha=0.1)

    yticks = np.linspace(min_value, max_value, num=5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(round(ytick, 2)) for ytick in yticks])
    ax.set_ylim([min_value, max_value])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.savefig(save_plot_name, bbox_inches="tight")


async def process_question(session, ques, ans1, ans2, category, rule_dict, args):
    rule = rule_dict[category]
    prompt = rule["prompt"]
    role = rule["role"]
    content = (
        f'[Context]\n{ques["context"]}\n\n'
        f'[Question]\n{ques["text"]}\n\n'
        f"[{role} 1]\n{ans1[args.gpt4_answer_col]}\n\n[End of {role} 1]\n\n"
        f"[{role} 2]\n{ans2[args.answer_col]}\n\n[End of {role} 2]\n\n"
        f"[System]\n{prompt}\n\n"
        f"If it is not relevant to the context, does not answer directly, or says the wrong thing, give it a low score.\n\n"
    )
    review = await get_eval(session, content, args.max_tokens)
    scores = parse_score(review)
    return {
        "question_id": ques["question_id"],
        "answer1_id": ans1.get("answer_id", ans1["question_id"]),
        "answer2_id": ans2.get("answer_id", ans2["question_id"]),
        "category": category,
        "content": review,
        "tuple": scores,
    }


async def main(args):
    async with aiohttp.ClientSession() as session:
        with open(os.path.expanduser(args.question)) as f_q, \
             open(os.path.expanduser(args.answer_list[0])) as f_ans1, \
             open(os.path.expanduser(args.answer_list[1])) as f_ans2, \
             open(os.path.expanduser(args.rule), "r") as f_rule, \
             open(os.path.expanduser(args.context)) as f_context:

            rule_dict = json.load(f_rule)
            context_list = [json.loads(line) for line in f_context]
            image_to_context = {context["image"]: context for context in context_list}

            cur_reviews = []
            if os.path.isfile(os.path.expanduser(args.output)):
                with open(os.path.expanduser(args.output)) as f_out:
                    cur_reviews = [json.loads(line) for line in f_out]

            with open(args.output, "a") as review_file:
                tasks = []
                cur_js_list = []
                for idx, (ques_js, ans1_js, ans2_js) in enumerate(
                    tqdm.tqdm(zip(f_q, f_ans1, f_ans2), total=TOTAL_QUESTIONS)
                ):
                    ques = json.loads(ques_js)
                    ans1 = json.loads(ans1_js)
                    ans2 = json.loads(ans2_js)
                    category = ques["category"]

                    if idx >= len(cur_reviews):
                        task = asyncio.create_task(process_question(session, ques, ans1, ans2, category, rule_dict, args))
                        tasks.append(task)
                    else:
                        print(f"Skipping question {idx} as we already have its review.")

                results = await asyncio.gather(*tasks)
                for result in results:
                    review_file.write(json.dumps(result, ensure_ascii=False) + "\n")

        name = args.output.split("/")[-1].split(".")[0]
        model_results_json = {name: args.output}
        model_results = load_model_results(model_results_json, TOTAL_QUESTIONS)
        plot_result(model_results, args.output.replace("json", "png"))
        print(f"Result: {model_results}")

        with open(args.output.replace(".json", "_summary.json"), "w") as f:
            f.write(json.dumps(model_results, indent=4))

        if args.is_upload_result:
            import wandb

            project_name = os.getenv("WANDB_PROJECT_NAME", "default-project")
            wandb.init(project=project_name, name=name)
            table = wandb.Table(columns=["Name", "mean", "conv", "detail", "complex"])
            for name, ret in model_results.items():
                table.add_data(
                    name,
                    (ret["conv"] + ret["detail"] + ret["complex"]) / 3,
                    ret["conv"],
                    ret["detail"],
                    ret["complex"],
                )
            wandb.log({"LB: LLaVA Bench Japanese": table})
            print("Uploaded results to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question", required=True, help="Path to the question file")
    parser.add_argument("-c", "--context", required=True, help="Path to the context file")
    parser.add_argument("-a", "--answer-list", nargs="+", default=[], help="Paths to the answer files")
    parser.add_argument("-r", "--rule", required=True, help="Path to the rule file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file")
    parser.add_argument("-gc", "--gpt4_answer_col", type=str, default="text", help="Column name for GPT-4 answers")
    parser.add_argument("-ac", "--answer_col", type=str, default="text", help="Column name for other answers")
    parser.add_argument("--is_upload_result", action="store_true", help="Upload results to wandb")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
