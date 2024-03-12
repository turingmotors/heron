import json

file_path = "./playground/data/llava-bench-in-the-wild/ja/output/exp001_answers.jsonl"


def remove_garbage_chars(line):
    data = json.loads(line)
    for key, value in data.items():
        if isinstance(value, str):
            value = value.replace("\n", "").replace("##human", "").replace("<image>", "")
            data[key] = value
    return json.dumps(data, ensure_ascii=False)


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        processed_lines = [remove_garbage_chars(line) for line in file]

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(processed_lines) + "\n")


process_file(file_path)
