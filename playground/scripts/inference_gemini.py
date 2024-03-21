import os
from pathlib import Path
import json
import google.generativeai as genai
from PIL import Image

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "top_p": 0.0,
}

# Gemini model
model_name = "models/gemini-1.0-pro-vision-latest"
model = genai.GenerativeModel(model_name, generation_config=generation_config)

# Input files
questions_file = "questions_ja.jsonl"
image_dir = Path("images")

# output file
output_file = "gemini_0314_ja.jsonl"

def load_data(file_path):
    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list

def generate_response(model, question, image):
    message = [question, image]
    response = model.generate_content(message)

    if hasattr(response._result, 'candidates') and response._result.candidates:
        candidate = response._result.candidates[0]
        answer = "".join(part.text for part in candidate.content.parts) if candidate.content.parts else "empty response"
    else:
        answer = "Blocked by the safety filter."

    return answer

def main():
    data_list = load_data(questions_file)
    
    with open(output_file, "w", encoding='utf-8') as out:
        for data in data_list:
            question_id = data["question_id"]
            question = data["text_ja"]
            image_path = data["image"]
            image = Image.open(image_dir / image_path)

            answer = generate_response(model, question, image)

            print("Question:", question)
            print("Answer:", answer, "\n")

            output = {
                "question_id": question_id,
                "images": image_path,
                "question": question,
                "answer_id": "",
                "model_id": model_name,
                "metadata": {},
                "text": answer
            }

            out.write(json.dumps(output, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
    