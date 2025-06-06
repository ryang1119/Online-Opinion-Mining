import json
import os
import argparse
import pickle
from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
from openai import OpenAI
from tqdm import tqdm
from eval_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoacu import A3CU
from evaluate import load

def get_output_schema():
    class Summary(BaseModel):
        summary: str
        
    return Summary.model_json_schema()

def replace_for_data_type(data_type):
    if data_type == 'youtube':
        return "youtube video thread comments"
    elif data_type == 'reddit':
        return "car community thread comments"
    elif data_type == 'blog':
        return "a blog post"
    elif data_type == 'review_site':
        return "car user review text"
    else:
        raise ValueError(f"Invalid data type {data_type}")

def generate_instance(prompt, text):
    return prompt + text

def init_args():
    parser = argparse.ArgumentParser(description="Feature-centric opinion mining (FOE) task inference script")
    parser.add_argument("--task", default='oig', type=str, help="The name of the task which inference is conducted on")
    parser.add_argument("--model_name", default='gpt-4o', type=str, help="The name of the model to be used for inference")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--port", default=8000, type=int, help="VLLM server port")

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
    if '/' in args.model_name:
        model_name = args.model_name.split('/')[1]
    else:
        model_name = args.model_name
            
    if 'deepseek' in model_name: 
        MAX_TOKENS = 4000
        SAFE_TOKENS = 3000
    else:
        MAX_TOKENS = 7500
        SAFE_TOKENS = 5000  
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    with open(f'{args.prompt_path}/oig_prompt.txt', 'r') as f:
        oig_instruction = f.read()
        
    with open(f'{args.data_path}/oomb_benchmark.pickle', 'rb') as f:
        data = pickle.load(f)

    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="-",
    )
    
    response_list = []

    for i in tqdm(range(len(data)), desc=f"Processing {model_name}"):
        oig_prompt = generate_instance(oig_instruction, data[i]['text'])
        input_text_ = generate_instance(oig_prompt, data[i]['text'])
        len_tokenize = len(tokenizer.tokenize(input_text_))
            
        if len_tokenize > MAX_TOKENS:
            print(f"Warning: The input text is too long. The length of the input text is {len_tokenize} tokens.")
            input_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(generate_instance(oig_prompt, data[i]['text']))[:SAFE_TOKENS])
            len_tokenize = len(tokenizer.tokenize(input_text))
        else:
            input_text = input_text_
        
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
    a3cu = A3CU(device=0)
    bertscore = load("bertscore")
    score = compute_oig_scores(response_list, gold_summary_list, bertscore, a3cu)
    print(f"Scores, {score}")

    # save scores
    with open(f"{args.output_dir}/{model_name}_inference_scores.json", 'w') as f:
        json.dump(score, f, indent=4)
        
        
if __name__ == "__main__":
    # logging.basicConfig(level=logging.NOTSET)
    args = init_args()
    print(f"Open-sourced model {args.model_name} {args.task} inference started")
    main(args)