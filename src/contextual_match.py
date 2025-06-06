import pickle
import os
from eval_utils import *
from openai import OpenAI
import json
from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
import argparse
from tqdm import tqdm
import pandas as pd 

class TuplePair(BaseModel):
    matched_pred_tuple: list[str, str, str]
    matched_gold_tuple: list[str, str, str]

class MatchedList(BaseModel):
    matched_tuple_pair: list[TuplePair]

def generate_gpt_match_prompt(prompt, gold, pred):
    return prompt + "\n\n<Gold>\n" + gold + "\n\n<Pred>\n" + pred
    
def init_args():
    parser = argparse.ArgumentParser(description="Feature-centric opinion mining (FOE) task inference script")
    parser.add_argument("--task", default='foe', type=str, help="The name of the task which inference is conducted on")
    parser.add_argument("--model_name", default='gpt-4o', type=str, help="The name of the model to be used for inference")

    args = parser.parse_args()

    args.output_dir =  f'../outputs/{args.task}/'
    args.data_path = f'../data/'
    args.prompt_path = f'../prompt/'

    return args


def main(args):
    with open(f'{args.prompt_path}/contextual_match_prompt.txt', 'r') as f:
        contextual_match_prompt = f.read()
        
    if '/' in args.model_name:
        model_name = args.model_name.split('/')[1]
    else:
        model_name = args.model_name
        
    API_KEY = ""
    client = OpenAI(api_key = API_KEY)
    print("*"*50)
    print(f"{model_name} {args.task} contextual match")
    print("*"*50)
    
    if os.path.exists(f"{args.output_dir}/{model_name}_inference_result.pickle"):
        outputs = pickle.load(open(f"{args.output_dir}/{model_name}_inference_result.pickle", "rb"))
    else:
        print(f"{model_name}_inference_result.pickle does not exist")
        exit()        
    
    pred_tuples, gold_tuples = outputs[0], outputs[1]
    matched_list = []
    for k in tqdm(range(len(pred_tuples))):
        attempt = 0
        success = False
        while attempt < 5 and not success:
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user", 
                            "content": generate_gpt_match_prompt(
                                contextual_match_prompt, 
                                str(gold_tuples[k]), 
                                str(pred_tuples[k])
                            )
                        },
                    ],
                    temperature=0.0,
                    response_format=MatchedList.model_json_schema()
                )
                result_json = json.loads(completion.choices[0].message.content)
                matched_list.append(result_json)
                success = True 
            except Exception as e:
                attempt += 1
                print(f"Retry {attempt} times")
                if attempt == 5:
                    print(f"Retry {attempt} times but failed")
                    matched_list.append({'matched_tuple_pair': [{'matched_pred_tuple': ['', '', ''], 'matched_gold_tuple': ['', '', '']}]})
    
    
    matched_tuple_pair_list = [x['matched_tuple_pair'] for x in matched_list]

    matched_pred_tuples, matched_gold_tuples = [], []

    for matched_tuple_pair in matched_tuple_pair_list:
        pred_temp, gold_temp = [], []
        for pair in matched_tuple_pair:
            pred_temp.append(pair['matched_pred_tuple'])
            gold_temp.append(pair['matched_gold_tuple'])
        matched_pred_tuples.append(pred_temp)
        matched_gold_tuples.append(gold_temp)

    score = compute_cm_scores(pred_tuples, gold_tuples, matched_pred_tuples, matched_gold_tuples)
    print(f"{model_name} Contextual Match:", score)
          
    with open(f"{args.output_dir}/{model_name}_cm_scores.json", 'w') as f:
        json.dump(score, f, indent=4)
    
if __name__ == "__main__":
    args = init_args()
    main(args)
