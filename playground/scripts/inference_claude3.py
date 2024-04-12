import base64
import io
import json
import os
from pathlib import Path

from anthropic import Anthropic
from PIL import Image

# Claude3 API 
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)
    
model_name = "claude-3-opus-20240229"

# Input files
questions_file = "questions_ja.jsonl"
image_dir = Path("images")

# output file
output_file = "claude3_0404_ja.jsonl"

def encode_image_to_base64(filepath):
    with Image.open(filepath) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

def process_question(client, question_data, image_dir):
    question_id = question_data["question_id"]
    image_category = question_data["image_category"]
    image_name = question_data["image"]
    prompt = question_data["text"]

    image_filepath = image_dir / image_name
    if not image_filepath.exists():
        print(f"Image file not found: {image_filepath}")
        return None

    image_data = encode_image_to_base64(image_filepath)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
            ],
        },
    ]

    response = client.messages.create(
        max_tokens=256,
        messages=messages,
        temperature=0,
        model=model_name,
    )

    decoded_text = response.content[0].text
    answer_id = response.id
    model = response.model

    return {
        "question_id": question_id,
        "images": image_name,
        "image_category": image_category,
        "prompt": prompt,
        "answer_id": answer_id,
        "model_id": model,
        "metadata": {},
        "text": decoded_text,
    }

def main():
    data_list = []
    with open(questions_file, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    with open(output_file, "w") as f:
        for question_data in data_list:
            output_data = process_question(client, question_data, image_dir)
            if output_data:
                print(output_data["text"])
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
