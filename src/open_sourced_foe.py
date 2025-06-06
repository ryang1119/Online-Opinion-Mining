import json
import os
import argparse
import pickle
from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_utils import *

def get_output_schema():
    class OpinionTuple(BaseModel):
        entity: str
        feature: str
        opinion: str

    class TupleList(BaseModel):
        opinion_tuple: list[OpinionTuple]
        
    json_schema = TupleList.model_json_schema()
    
    return json_schema

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
    return prompt + text + "\n\n<Tuple List>\n"

def init_args():
    parser = argparse.ArgumentParser(description="Feature-centric opinion mining (FOE) task inference script")
    parser.add_argument("--task", default='foe', type=str, help="The name of the task which inference is conducted on")
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
    
    with open(f'{args.prompt_path}/foe_prompt.txt', 'r') as f:
        foe_instruction = f.read()
        
    with open(f'{args.data_path}/oomb_benchmark.pickle', 'rb') as f:
        data = pickle.load(f)

    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="-",
    )
    
    response_list = []

    for i in tqdm(range(len(data)), desc=f"Processing {model_name}"):
        foe_prompt = foe_instruction.replace('{REPLACE}', replace_for_data_type(data[i]['content_type']))
        input_text_ = generate_instance(foe_prompt, data[i]['text'])
        len_tokenize = len(tokenizer.tokenize(input_text_))
        
        if len_tokenize > MAX_TOKENS:
            print(f"Warning: The input text is too long. The length of the input text is {len_tokenize} tokens.")
            input_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(generate_instance(foe_prompt, data[i]['text']))[:SAFE_TOKENS])
        else:
            input_text = input_text_
        
        attempt = 0
        max_attempts = 5
        success = False
        result_json = {"opinion_tuple": []} 

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
                raw_content = completion.choices[0].message.content
                try:
                    result_json = json.loads(raw_content)
                except json.JSONDecodeError:
                    corrected_content = raw_content.replace("'", '"').strip()
                    result_json = json.loads(corrected_content)

                success = True  
                print(f"Output: {raw_content[:100]}...")

            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt >= max_attempts:
                    print("Max attempts reached. Skipping this instance.")
                    result_json = {"opinion_tuple": []}
                    break

        response_list.append(result_json.get('opinion_tuple', []))

    pred_tuple = []
    for i in range(len(response_list)):
        if len(response_list[i]) == 0:
            pred_tuple.append([])
            continue
        temp_tuple = [(str(item['entity']).lower(), str(item['feature']).lower(), str(item['opinion']).lower()) for item in response_list[i]]
        temp_tuple = list(set(temp_tuple))
        pred_tuple.append(temp_tuple)

    gold_tuple = [
        [
            (str(t['entity']).lower(), str(t['feature']).lower(), str(t['opinion']).lower())
            for t in item['tuple']
        ]
        for item in data
    ]

    # save inference results
    with open(f"{args.output_dir}/{model_name}_inference_result.pickle", 'wb') as f:
        pickle.dump((pred_tuple, gold_tuple), f)
    
    print(f"{model_name} inference completed")

    st_model = SentenceTransformer("all-MiniLM-L6-v2").to('cuda:0')
    # Compute scores
    score = compute_foe_scores(pred_tuple, gold_tuple, st_model)
    print(f"Scores for {model_name}:", score)

    with open(f"{args.output_dir}/{model_name}_inference_scores.json", 'w') as f:
        json.dump(score, f, indent=4)
    
if __name__ == "__main__":
    #logging.basicConfig(level=logging.NOTSET)
    args = init_args()
    print(f"Open-sourced model {args.model_name} {args.task} inference started")
    main(args)