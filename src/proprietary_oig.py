import json
import os
import argparse
import logging
import pickle
# from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_utils import *
from autoacu import A3CU
from evaluate import load
import instructor
from anthropic import Anthropic
from pydantic import BaseModel

class Summary(BaseModel):
    summary: str

def replace_for_data_type(data_type):
    if data_type == 'youtube':
        return "youtube video thread comments"
    elif data_type == 'reddit':
        return "car community thread comments"
    elif data_type == 'blog':
        return "blog post"
    elif data_type == 'review_site':
        return "car user review text"
    else:
        raise ValueError(f"Invalid data type {data_type}")

def generate_instance(prompt, text):
    return prompt + text

def run_gpt_model(model_name, data, oig_instruction, client, args, a3cu, bertscore):
    from tooldantic import OpenAiResponseFormatBaseModel as BaseModel

    def get_output_schema():
        class Summary(BaseModel):
            summary: str

        json_schema = Summary.model_json_schema()
        
        return json_schema
    
    response_list = []

    for i in tqdm(range(len(data)), desc=f"Processing {model_name}"):
        oig_prompt = oig_instruction.replace('{REPLACE}', replace_for_data_type(data[i]['content_type']))
        input_text = generate_instance(oig_prompt, data[i]['text'])

        attempt = 0
        max_attempts = 5
        success = False
        result_json = {"summary": ""}
        raw_response = ""

        while attempt < max_attempts and not success:
            try:
                completion = client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.0,
                    response_format=get_output_schema(),
                )
                raw_response = completion.choices[0].message.content
                
                try:
                    result_json = json.loads(raw_response)
                except json.JSONDecodeError:
                    corrected_content = raw_response.replace("'", '"').strip()
                    result_json = json.loads(corrected_content)
                
                success = True  
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed for input {input_text} with error: {e}")
                if attempt >= max_attempts:
                    print("Max attempts reached. Skipping this instance.")
                    result_json = {"summary": ""}
                    raw_response = ""
                    break
        
        response_list.append(result_json.get('summary', ""))
    
    with open(f"{args.output_dir}/{model_name}_inference_result.pickle", 'wb') as f:
        pickle.dump(response_list, f)
    
    gold_summary_list = [item['summary'] for item in data]
    score = compute_oig_scores(response_list, gold_summary_list, bertscore, a3cu)
    print(f"Scores for {model_name}:", score)
    
    # Save scores
    with open(f"{args.output_dir}/{model_name}_inference_scores.json", 'w') as f:
        json.dump(score, f, indent=4)
    
    return model_name, score


def run_claude_model(model_name, data, oig_instruction, client, args, a3cu, bertscore):
    from pydantic import BaseModel

    class Summary(BaseModel):
        summary: str

    response_list = []

    for i in tqdm(range(len(data)), desc=f"Processing {model_name}"):

        oig_prompt = oig_instruction.replace('{REPLACE}', replace_for_data_type(data[i]['content_type']))
        input_text = generate_instance(oig_prompt, data[i]['text'])

        attempt = 0
        max_attempts = 5
        success = False
        summary = ""
        
        while attempt < max_attempts and not success:
            try:
                message = client.messages.create(
                    model=model_name,
                    max_tokens=8192,
                    temperature=0.0,
                    system="",
                    messages=[
                        {"role": "user", "content": input_text},
                    ],
                    response_model=Summary,
                )
                summary = message.summary
                success = True
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed for input {input_text} with error: {e}")
                if attempt >= max_attempts:
                    print("Max attempts reached. Skipping this instance.")
                    summary = ""
                    break

        print(f"{model_name} Output: {summary[:100]}...")
        
        if not summary:
            response_list.append("")
        else:
            response_list.append(summary)

    with open(f"{args.output_dir}/{model_name}_inference_result.pickle", 'wb') as f:
        pickle.dump(response_list, f)
    
    gold_summary_list = [item['summary'] for item in data]
    score = compute_oig_scores(response_list, gold_summary_list, bertscore, a3cu)
    print(f"Scores for {model_name}:", score)
    
    # Save scores
    with open(f"{args.output_dir}/{model_name}_inference_scores.json", 'w') as f:
        json.dump(score, f, indent=4)
    
    return model_name, score

def init_args():
    parser = argparse.ArgumentParser(description="Feature-centric opinion mining (FOE) task inference script")
    parser.add_argument("--task", default='oig', type=str, help="The name of the task which inference is conducted on")
    parser.add_argument("--model_name", default='gpt-4o', type=str, help="The name of the model to be used for inference")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    args = parser.parse_args()

    args.output_dir =  f'../outputs/{args.task}/'
    args.data_path = f'../data/'
    args.prompt_path = f'../prompt/'
    
    if not os.path.exists(f'../outputs'):
        os.mkdir(f'../outputs')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def main(args):
    with open(f'{args.prompt_path}/oig_prompt.txt', 'r') as f:
        oig_instruction = f.read()
          
    a3cu = A3CU(device=0)
    bertscore = load("bertscore")

    # data load
    with open(f'{args.data_path}/oomb_benchmark.pickle', 'rb') as f:
        data = pickle.load(f)

    if 'gpt' in args.model_name:
        API_KEY = ""
        client = OpenAI(api_key = API_KEY)
        model_name, model_score = run_gpt_model(args.model_name, data, oig_instruction, client, args, a3cu, bertscore)
        print(f"Scores for {model_name}:", model_score)
        
    else:
        API_KEY = ""
        client = instructor.from_anthropic(Anthropic(api_key=API_KEY))
        model_name, model_score = run_claude_model(args.model_name, data, oig_instruction, client, args, a3cu, bertscore)
        print(f"Scores for {model_name}:", model_score)

if __name__ == "__main__":
    args = init_args()
    print(f"Proprietary LLMs {args.model_name} {args.task} inference started")
    main(args)