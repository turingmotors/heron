import argparse
import asyncio
import json
import os
from collections import defaultdict

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import tqdm

NUM_SECONDS_TO_SLEEP = 0.5


async def get_eval(session, content: str, max_tokens: int):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY') }",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4-0613",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and precise assistant for checking the quality of the answer.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    while True:
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 429:
                    await asyncio.sleep(NUM_SECONDS_TO_SLEEP)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print("error", review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print("error", review)
        return [-1, -1]


def load_jsonl(path, num):
    scores = defaultdict(list)
    for i, line in enumerate(open(path)):
        if i > num:
            break
        d = json.loads(line)
        scores[d["category"]].append(d["tuple"][1])
        scores[d["category"] + "_ref"].append(d["tuple"][0])
    return scores


def load_model_results(model_results, num=90):
    results = {}
    for model_name, result_path in model_results.items():
        scores = load_jsonl(result_path, num)
        result = {}
        for c, s in scores.items():
            if "ref" not in c:
                # 比較対象とターゲットのスコアの平均値の比率をllava-benchのスコアとする
                rel_score = 100 * np.mean(s) / np.mean(scores[c + "_ref"])
                result[c] = rel_score
        results[model_name] = result
    return results


def plot_result(model_results, save_plot_name, min_value=0, max_value=110):
    # データの設定
    labels = list(model_results[list(model_results.keys())[0]].keys())
    model_scores = {}
    for model_name, result in model_results.items():
        model_scores[model_name] = [max(0, result[label]) for label in labels]
        model_scores[model_name] += model_scores[model_name][:1]

    # レーダーチャートを描画するための角度を計算
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # 最初の角度をリストの最後に追加して円を閉じる

    # レーダーチャートの描画
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    for i, (model_name, score) in enumerate(model_scores.items()):
        ax.plot(angles, score, color=colorlist[i % len(colorlist)], linewidth=2, label=model_name)
        ax.fill(angles, score, color=colorlist[i % len(colorlist)], alpha=0.1)

    # グラフの見た目を調整
    # メモリの追加
    yticks = np.linspace(min_value, max_value, num=5)  # min_valueからmax_valueまでを5等分
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(round(ytick, 2)) for ytick in yticks])  # メモリに表示する値（小数点第2位まで）

    # ax.set_yticklabels([])
    ax.set_ylim([min_value, max_value])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    # plt.show()
    plt.savefig(save_plot_name)


async def main(args):
    async with aiohttp.ClientSession() as session:
        f_q = open(os.path.expanduser(args.question))
        f_ans1 = open(os.path.expanduser(args.answer_list[0]))
        f_ans2 = open(os.path.expanduser(args.answer_list[1]))
        rule_dict = json.load(open(os.path.expanduser(args.rule), "r"))

        if os.path.isfile(os.path.expanduser(args.output)):
            cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
        else:
            cur_reviews = []

        review_file = open(f"{args.output}", "a")

        context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
        image_to_context = {context["image"]: context for context in context_list}

        tasks = []
        cur_js_list = []
        for idx, (ques_js, ans1_js, ans2_js) in enumerate(
            tqdm.tqdm(zip(f_q, f_ans1, f_ans2), total=90)
        ):
            ques = json.loads(ques_js)
            ans1 = json.loads(ans1_js)
            ans2 = json.loads(ans2_js)

            inst = image_to_context[ques["image"]]
            cap_str = "\n".join(inst["captions"])
            box_str = "\n".join(
                [f'{instance["category"]}: {instance["bbox"]}' for instance in inst["instances"]]
            )

            category = json.loads(ques_js)["category"]
            if category in rule_dict:
                rule = rule_dict[category]
            else:
                assert False, f"Visual QA category not found in rule file: {category}."
            prompt = rule["prompt"]
            role = rule["role"]
            content = (
                f"[Context]\n{cap_str}\n\n{box_str}\n\n"
                f'[Question]\n{ques["text"]}\n\n'
                f"[{role} 1]\n{ans1[args.gpt4_answer_col]}\n\n[End of {role} 1]\n\n"
                f"[{role} 2]\n{ans2[args.answer_col]}\n\n[End of {role} 2]\n\n"
                f"[System]\n{prompt}\n\n"
            )
            cur_js = {
                "id": idx + 1,
                "question_id": ques["question_id"],
                "answer1_id": ans1.get("answer_id", ans1["question_id"]),
                "answer2_id": ans2.get("answer_id", ans2["question_id"]),
                "category": category,
            }
            if idx >= len(cur_reviews):
                task = asyncio.create_task(get_eval(session, content, args.max_tokens))
                tasks.append(task)
                cur_js_list.append(cur_js)
            else:
                print(f"Skipping {idx} as we already have it.")

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Process results and write to file as before
        for result, cur_js in zip(results, cur_js_list):
            review = result
            scores = parse_score(review)
            # Assuming `cur_js` is prepared as before:
            cur_js["content"] = review
            cur_js["tuple"] = scores
            review_file.write(json.dumps(cur_js, ensure_ascii=False) + "\n")
            review_file.flush()

        review_file.close()

        name = args.output.split("/")[-1].split(".")[0]
        model_results_json = {
            name: args.output,
        }
        model_results = load_model_results(model_results_json)
        plot_result(model_results, args.output.replace("json", "png"), 0, 110)
        print(f"result: {model_results}")
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
            print("Upload results to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question")
    parser.add_argument("-c", "--context")
    parser.add_argument("-a", "--answer-list", nargs="+", default=[])
    parser.add_argument("-r", "--rule")
    parser.add_argument("-o", "--output")
    parser.add_argument("-gc", "--gpt4_answer_col", type=str, default="text_JA")
    parser.add_argument("-ac", "--answer_col", type=str, default="answer")
    parser.add_argument("--is_upload_result", action="store_true")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
