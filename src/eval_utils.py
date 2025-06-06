from difflib import SequenceMatcher
import torch
from tqdm import tqdm
import statistics
from rouge import Rouge

def get_lexical_similarities(pred_pt, gold_pt):
    similarities = []
    for pred_tuple in tqdm(pred_pt, desc="Processing predicted tuples"): 
        row = []
        for gold_tuple in gold_pt:
            if len(pred_tuple) != len(gold_tuple):
                row.append([0.0])
            else:
                sims = [SequenceMatcher(None, p_comp, g_comp).ratio() for p_comp, g_comp in zip(pred_tuple, gold_tuple)]
                row.append(sims)
        similarities.append(row)
    return similarities

def encode_text(text, model):
    return model.encode(text, convert_to_tensor=True, show_progress_bar=False)

def cosine_similarity(vec1, vec2):
    numerator = torch.sum(vec1 * vec2)
    denominator = torch.sqrt(torch.sum(vec1 * vec1)) * torch.sqrt(torch.sum(vec2 * vec2))
    return (numerator / denominator).item()

def get_semantic_similarities(pred_pt, gold_pt, model): 
    similarities = []
    for pred_tuple in tqdm(pred_pt, desc="Processing predicted tuples"):
        row = []
        for gold_tuple in gold_pt:
            if len(pred_tuple) != len(gold_tuple):
                row.append([0.0])
            else:
                sims = []
                for p_comp, g_comp in zip(pred_tuple, gold_tuple):
                    emb_p = encode_text(p_comp, model)
                    emb_g = encode_text(g_comp, model)
                    sims.append(cosine_similarity(emb_p, emb_g))
                row.append(sims)
        similarities.append(row)
    return similarities

def compute_f1_for_threshold(pred_pt, gold_pt, all_sims, threshold):
    n_pred, n_gold = 0, 0
    n_tp, unique_n_tp = 0, 0
    
    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
    
    for i in tqdm(range(len(pred_pt)), desc=f"Calculating F1 for threshold={threshold}"):
        sims_for_sample = all_sims[i]
        gold_matched_indices = set()
        
        for pred_idx in range(len(pred_pt[i])):
            matched = False
            for gold_idx in range(len(gold_pt[i])):
                comp_sims = sims_for_sample[pred_idx][gold_idx]
                if all(sim >= threshold for sim in comp_sims):
                    matched = True
                    gold_matched_indices.add(gold_idx)
                    break
            if matched:
                n_tp += 1
        unique_n_tp += len(gold_matched_indices)
    
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(unique_n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2)
    }

def compute_exact_f1_scores(pred_pt, gold_pt):
    n_tp, n_gold, n_pred = 0, 0, 0
    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1': round(f1 * 100, 2)
    }

def compute_foe_scores(preds, golds, model):
    results = {}
    results["exact_match"] = compute_exact_f1_scores(preds, golds)
    print(f"Exact Match => {results['exact_match']}")
    all_lexical_sims = []
    all_semantic_sims = []
    for i in tqdm(range(len(preds)), desc="Gathering similarities"):
        all_lexical_sims.append(get_lexical_similarities(preds[i], golds[i]))
        all_semantic_sims.append(get_semantic_similarities(preds[i], golds[i], model))
    thresholds = 0.7
    
    results[f"lexical_relaxed_match"] = compute_f1_for_threshold(preds, golds, all_lexical_sims, thresholds)
    results[f"semantic_relaxed_match"] = compute_f1_for_threshold(preds, golds, all_semantic_sims, thresholds)
    
    print(f"Lexical Relaxed Match => {results['lexical_relaxed_match']}")
    print(f"Semantic_relaxed_match Relaxed Match  => {results['semantic_relaxed_match']}")
    
    return results

def compute_cm_scores(pred_seqs, gold_seqs, matched_pred_tuple_list, matched_gold_tuple_list):

    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        all_labels.append(gold_seqs[i])
        all_preds.append(pred_seqs[i])

    n_gold, n_pred = 0, 0
    n_tp, uniqe_n_tp = 0, 0
    
    for i in range(num_samples):
        n_gold += len(all_labels[i])
        n_pred += len(all_preds[i])
        n_tp += len(matched_pred_tuple_list[i])
        
        unique_tuples = set(tuple(item) for item in matched_gold_tuple_list[i])
        unique_list = [list(item) for item in unique_tuples]
        uniqe_n_tp += len(unique_list)

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(uniqe_n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    cm_scores = {
        'contextual_match': {
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }
    }
    return cm_scores

def compute_a3cu(preds, golds, a3cu):
    pre, rec, f1 = a3cu.score(
        references=golds,
        candidates=preds,
        batch_size=16,
        output_path=None,
    )
    return {
        'precision': round(statistics.mean(pre) * 100, 2),
        'recall': round(statistics.mean(rec) * 100, 2),
        'f1': round(statistics.mean(f1) * 100, 2)
    }
    
def compute_rouge(preds, golds):
    rouge = Rouge()
    rouge_1_scores, rouge_2_scores, rouge_l_scores = [], [], []

    for pred, ref in zip(preds, golds):
        if not pred.strip():
            print("Skipping empty prediction")
            continue

        scores = rouge.get_scores(pred, ref)[0]
        rouge_1_scores.append(scores['rouge-1']['f'])
        rouge_2_scores.append(scores['rouge-2']['f'])
        rouge_l_scores.append(scores['rouge-l']['f'])

    if not rouge_1_scores:  
        return {"r-1": 0.0, "r-2": 0.0, "r-L": 0.0}

    return {
        "r-1": round(statistics.mean(rouge_1_scores) * 100, 2),
        "r-2": round(statistics.mean(rouge_2_scores) * 100, 2),
        "r-L": round(statistics.mean(rouge_l_scores) * 100, 2)
    }
    
def compute_bert_score(preds, golds, bertscore):
    results = bertscore.compute(predictions=preds, references=golds, lang="en")
    return {
        'precision': round(statistics.mean(results['precision']) * 100, 2),
        'recall': round(statistics.mean(results['recall']) * 100, 2),
        'f1': round(statistics.mean(results['f1']) * 100, 2)
    }
    
def compute_oig_scores(preds, golds, bertscore, a3cu):
    a3cu_score = compute_a3cu(preds, golds, a3cu)
    rouge_score = compute_rouge(preds, golds)
    bert_score = compute_bert_score(preds, golds, bertscore)
    
    return {
        'rouge': rouge_score,
        'bert_score': bert_score,
        'a3cu': a3cu_score
    }