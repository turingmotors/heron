import openai
from openai import OpenAI
import json
import jsonlines
import os
import base64
import time
import uuid
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return None

def process_row(row, writer):
    base64_image = encode_image(f"./images/{row['image']}".replace('png', 'jpg'))
    if base64_image is None:
        return

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row['text']},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=512,
    )

    result = {
        "question_id": row["question_id"],
        "images": row["image"],
        "categories": row["category"],
        "image_categories": row["image_category"],
        "prompt": row["text"],
        "answer_id": "",
        "model_id": "gpt-4-vision-preview",
        "metadata": {},
        "text": response.choices[0].message.content
    }
    writer.write(result)

def main():
    with open('questions_ja.jsonl', 'r') as f:
        data = [json.loads(l) for l in f.readlines()]

    with jsonlines.open('gpt4v_0314_ja.jsonl', mode='w') as writer:
        for row in tqdm(data):
            process_row(row, writer)
            time.sleep(1)

if __name__ == "__main__":
    main()
