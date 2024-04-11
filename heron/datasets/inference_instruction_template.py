# Copyright 2023 Turing Inc. Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def base_instruction(content):
    prompt = f"##human: {content}\n##gpt: "
    return prompt

def none_instruction(content):
    prompt = f"{content}\n"
    return prompt

def llama2_instruction(content, is_system_message):
    if is_system_message:
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n"
        prompt = f"[INST] <<SYS>>\n{system_prompt}<</SYS>>\n\n{content} [/INST] "
        return prompt
    else:
        prompt = f"[INST] {content} [/INST] "
        return prompt

def tinyllama_instruction(content, tokenizer, is_system_message):
    if is_system_message:
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."
        prompt = f"<|system|>\n{system_prompt}{tokenizer.eos_token}\n<|user|>\n{content}{tokenizer.eos_token}\n<|assistant|>\n"
        return prompt
    else:   
        prompt = f"<|user|>\n{content}{tokenizer.eos_token}\n<|assistant|>\n"
        return prompt

def mistral_instruction(content):
    prompt = f"[INST] {content} [/INST] "
    return prompt

def commandr_instruction(content):    
    prompt = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    return prompt


def add_inference_instruction_template(content, tokenizer, instruction_template_type, is_system_message):
    if instruction_template_type == "llama2":
        prompt = llama2_instruction(content, is_system_message)
        return prompt
    elif instruction_template_type in ("mistral", "mixtral"):
        prompt = mistral_instruction(content)
        return prompt
    elif instruction_template_type == "command-r":
        prompt = commandr_instruction(content)
        return prompt
    elif instruction_template_type == "tinyllama":
        prompt = tinyllama_instruction(content, tokenizer, is_system_message)
        return prompt
    elif instruction_template_type == "none":
        prompt = none_instruction(content)
        return prompt
    else:
        prompt = base_instruction(content)
        return prompt
