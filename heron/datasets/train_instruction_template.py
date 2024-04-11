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

def base_instruction(agent, tokenizer):
    if agent == "gpt":
        agent_prompt = ""
        next_agent_prompt = f"{tokenizer.eos_token}"
    elif agent == "human":
        agent_prompt = "##human: "
        next_agent_prompt = "\n##gpt: "
    return agent_prompt, next_agent_prompt

def none_instruction(agent, tokenizer):
    if agent == "gpt":
        agent_prompt = ""
        next_agent_prompt = f"{tokenizer.eos_token}"
    elif agent == "human":
        agent_prompt = ""
        next_agent_prompt = ""
    return agent_prompt, next_agent_prompt

def llama2_instruction(agent, tokenizer, is_system_message):
    if is_system_message:
        if agent == "gpt":
            agent_prompt = ""
            next_agent_prompt = f"{tokenizer.eos_token}"
        elif agent == "human":
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n"
            agent_prompt = f"[INST] <<SYS>>\n{system_prompt}<</SYS>>\n\n"
            next_agent_prompt = " [/INST] "
    else:
        if agent == "gpt":
            agent_prompt = ""
            next_agent_prompt = f"{tokenizer.eos_token}"
        elif agent == "human":
            agent_prompt = "[INST] "
            next_agent_prompt = " [/INST] "
    return agent_prompt, next_agent_prompt

def tinyllama_instruction(agent, tokenizer, is_system_message):
    if is_system_message:
        if agent == "gpt":
            agent_prompt = ""
            next_agent_prompt = f"{tokenizer.eos_token}"
        elif agent == "human":
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."
            agent_prompt = f"<|system|>\n{system_prompt}{tokenizer.eos_token}\n<|user|>\n"
            next_agent_prompt = f"{tokenizer.eos_token}\n<|assistant|>\n"
    else:
        if agent == "gpt":
            agent_prompt = ""
            next_agent_prompt = f"{tokenizer.eos_token}"
        elif agent == "human":
            agent_prompt = f"<|user|>\n"
            next_agent_prompt = f"{tokenizer.eos_token}\n<|assistant|>\n"
    return agent_prompt, next_agent_prompt

def mistral_instruction(agent, tokenizer):
    if agent == "gpt":
        agent_prompt = ""
        next_agent_prompt = f"{tokenizer.eos_token}"
    elif agent == "human":
        agent_prompt = "[INST] "
        next_agent_prompt = " [/INST] "
    return agent_prompt, next_agent_prompt

def commandr_instruction(agent, tokenizer):
    if agent == "gpt":
        agent_prompt = ""
        next_agent_prompt = f"{tokenizer.eos_token}"
    elif agent == "human":
        agent_prompt = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
        next_agent_prompt = "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    return agent_prompt, next_agent_prompt


def add_train_instruction_template(agent, tokenizer, instruction_template_type, is_system_message):
    if instruction_template_type == "llama2":
        agent_prompt, next_agent_prompt = llama2_instruction(agent, tokenizer, is_system_message)
        return agent_prompt, next_agent_prompt
    elif instruction_template_type in ("mistral", "mixtral"):
        agent_prompt, next_agent_prompt = mistral_instruction(agent, tokenizer)
        return agent_prompt, next_agent_prompt
    elif instruction_template_type == "command-r":
        agent_prompt, next_agent_prompt = commandr_instruction(agent, tokenizer)
        return agent_prompt, next_agent_prompt
    elif instruction_template_type == "tinyllama":
        agent_prompt, next_agent_prompt = tinyllama_instruction(agent, tokenizer, is_system_message)
        return agent_prompt, next_agent_prompt
    elif instruction_template_type == "none":
        agent_prompt, next_agent_prompt = none_instruction(agent, tokenizer)
        return agent_prompt, next_agent_prompt
    else:
        agent_prompt, next_agent_prompt = base_instruction(agent, tokenizer)
        return agent_prompt, next_agent_prompt
