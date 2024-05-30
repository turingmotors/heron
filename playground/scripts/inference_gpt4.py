import json
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_questions(data):
    return [item.get('text') for item in data if item.get('text')]

def extract_contexts(data):
    return {item['image']: item['caption_jp'] for item in data if 'image' in item and 'caption_jp' in item}

def generate_answer(context, question):
    prompt = f"""
    あなたはAIビジュアルアシスタントであり、一枚の画像を見ています。「コンテキスト: {context}」はあなたが見ている画像に対する説明文です。
    実際に画像を見ているかのように、「質問: {question}」の質問に最大3文程度で簡潔かつ明瞭に答えてください。
    実際に画像を見ているという前提を遵守し、「画像を見ることができません」のような回答や曖昧な回答はしてはいけません。
    オブジェクトの種類、数、アクション、位置、オブジェクト間の相対位置などを厳密に含めて回答をしてください。
    """
    messages = [{"role": "system", "content": prompt}]
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=messages,
        top_p=0.0,
    )
    return completion.choices[0].message.content

def process_question(question_data, context):
    question = question_data["text"]
    try:
        answer = generate_answer(context, question)
        return {
            "question_id": question_data["question_id"],
            "images": question_data["image"],
            "image_category": question_data["image_category"],
            "prompt": question_data["text"],
            "answer_id": "",
            "model_id": "gpt-4-0125-preview",
            "text": answer,
        }
    except Exception as e:
        print(f"Error processing question {question}: {e}")
        return None

def process_questions(questions_data, contexts_data, output_path):
    contexts = extract_contexts(contexts_data)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for question_data in questions_data:
            image = question_data["image"]
            context = contexts.get(image)
            if context:
                answer_entry = process_question(question_data, context)
                if answer_entry:
                    output_file.write(json.dumps(answer_entry, ensure_ascii=False) + '\n')
            else:
                print(f"Context not found for image {image}")

def main():
    questions_path = './questions_ja.jsonl'
    contexts_path = './context_ja.jsonl'
    output_path = './answers_gpt4.jsonl'

    questions_data = load_jsonl(questions_path)
    contexts_data = load_jsonl(contexts_path)

    print("----- Start -----")
    process_questions(questions_data, contexts_data, output_path)

if __name__ == "__main__":
    main()
