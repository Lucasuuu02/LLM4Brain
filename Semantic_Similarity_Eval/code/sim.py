import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer


import numpy as np
from scipy.spatial.distance import cosine

def compute_sentence_similarity(ref_emb, cand_emb):
    """
    Calculate similarity between reference embedding and candidate embedding
    
    Args:
        ref_emb: (layers, size)
        cand_emb: (layers, size)
    
    Returns:
        float: Average similarity (across all layers)
    """
    layers = ref_emb.shape[0]
    layer_sims = []
    
    for layer in range(layers):
        # Calculate cosine similarity (1 - cosine distance)
        sim = 1 - cosine(ref_emb[layer], cand_emb[layer])
        layer_sims.append(sim)
    
    return np.mean(layer_sims)  # Average similarity across all layers

def evaluate_model_performance(model_embs):
    """
    Evaluate model performance:
    - Calculate similarity between reference (sentence 1) vs. candidates (sentences 2-6)
    - Return average similarity as model score
    
    Args:
        model_embs: Embedding matrix of shape (1577, 6, layers, size)
    
    Returns:
        float: Model performance score (average similarity of 5 candidate sentences)
    """
    ref_emb = model_embs[0, 0]  # Embedding of first sentence (layers, size)
    cand_embs = model_embs[0, 1:6]  # Embeddings of sentences 2-6 (5, layers, size)
    
    sim_scores = []
    for cand_emb in cand_embs:
        sim = compute_sentence_similarity(ref_emb, cand_emb)
        sim_scores.append(sim)
    
    return np.mean(sim_scores)  # Average similarity of 5 candidate sentences

class get_simlirities():
    def __init__(self,cls_use):
        self.cls_use = cls_use
    def get_sentence_vector(self,embedding):
        """Extract embedding from specified layer and generate sentence vector"""
        last_layer = embedding[:, :]  # Extract specified layer, shape (tokens, size)
        if self.cls_use:
            return last_layer[0]
        else:
            return np.mean(last_layer, axis=0)

    def simlirities(self,c_embedding,e_embedding):
        c_vector = self.get_sentence_vector(c_embedding)  # Use average pooling
        e_vector = self.get_sentence_vector(e_embedding)
        sim = cosine_similarity([c_vector], [e_vector])[0][0]
        return sim

def tokenizer_has_cls(model_path,model):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path,model),trust_remote_code=True)
    return tokenizer.cls_token is not None

if __name__ == '__main__':
    new_cn_en_dict = {0: 0, 1: 1, 2: 2, 10: 7, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 
                      19: 17, 23: 19, 24: 20, 25: 21, 26: 22, 28: 25, 38: 32, 39: 33, 40: 34, 41: 35, 
                      42: 36, 43: 37, 45: 39, 46: 40, 48: 42, 49: 43, 51: 44, 54: 46, 55: 47, 56: 48, 57: 49, 
                      59: 50, 60: 51, 61: 52, 62: 53, 71: 60, 72: 61, 73: 62, 74: 63, 77: 65, 78: 66, 82: 71, 
                      83: 72, 84: 73, 85: 74, 88: 77, 89: 78, 90: 79, 91: 80, 92: 81, 93: 83, 94: 83, 95: 84, 
                      96: 85, 97: 86, 98: 88, 99: 89, 100: 90, 103: 92, 105: 97, 106: 98, 107: 99, 108: 100, 
                      109: 101, 110: 102, 111: 103, 112: 104, 115: 106, 116: 107, 117: 108, 120: 110, 124: 113, 
                      126: 115, 129: 117, 130: 118, 133: 121, 135: 122, 139: 126, 140: 127, 141: 128, 142: 129, 
                      143: 130, 144: 131, 147: 133, 153: 138, 154: 139, 157: 141, 158: 143, 161: 151, 67: 152, 
                      168: 155, 172: 157, 176: 159, 177: 161, 178: 160, 179: 162, 180: 163, 183: 165, 184: 166, 
                      185: 167, 186: 168, 187: 169, 188: 170, 191: 172, 192: 173, 196: 177, 205: 183, 206: 184, 
                      207: 185, 208: 186, 209: 187, 212: 190}
    # ========== Configuration Paths ==========
    # Please modify the following paths according to your actual directory structure
    root_path = './path/to/your/data'  # Change to your data root directory path
    model_path = './path/to/your/models'  # Change to your model root directory path
    
    save_path = './all_sem_result_eval_scheme'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Model list: Select models you need to evaluate
    model_list =  ["Baichuan2-7B-Chat","gemma-2-9b","Llama-3.1-8B","opt-6.7b","Qwen2.5-7B",
    "Baichuan-7B","gemma-2-9b-it","Llama-3.1-8B-Instruct","Qwen2.5-7B-Instruct",
    "bert-base-uncased","glm-4-9b-chat-hf","Mistral-7B-Instruct-v0.3",
    "DeepSeek-R1-Distill-Qwen-7B","glm-4-9b-hf","Mistral-7B-v0.3"]
    
    # Embedding data paths
    cn_em_root_path = os.path.join(root_path,'emb_data/scheme1')
    en_em_root_path = os.path.join(root_path,'emb_data/choices_emb_all/scheme1')
    columns = ['simA', 'simB', 'simC', 'simD', 'simE', 'result']
    num_file = len(os.listdir(os.path.join(en_em_root_path,'A/embedding_Baichuan2-7B-Chat')))
    # Iterate through model list
    reslut_dict ={}
    for mo in model_list:
        # Embedding folder name
        cn_folder = "embedding_CN_{}_scheme1".format(mo.split('/')[-1])
        en_folder = "embedding_{}".format(mo.split('/')[-1])
        en_list = ['A','B','C','D','E']
        SIM_cl = get_simlirities(tokenizer_has_cls(model_path,mo))
        tb = []
        all_embeddings_list = []
        for key,value in tqdm(new_cn_en_dict.items(),total=len(new_cn_en_dict),desc='Processing'):
            cn_em_idx = str(key)+'.npy'
            en_em_idx = str(value)+'.npy'
            embeddings_list = []
            cn_file = os.path.join(cn_em_root_path,cn_folder,cn_em_idx)
            cn_embed = np.load(cn_file)
            row=[]
            for en_c in en_list:
                en_file = os.path.join(en_em_root_path,en_c,en_folder,en_em_idx)
                en_embed = np.load(en_file)
                #print(en_embed.shape)
                embeddings_list.append(en_embed)
            embeddings_list.append(cn_embed)
            all_embeddings_list.append(embeddings_list)
        model_result = evaluate_model_performance(np.array(all_embeddings_list))
        reslut_dict[mo] = model_result
        # Convert to serializable format
    serializable_dict = {k: float(v) for k, v in reslut_dict.items()}
        # Save as JSON file
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(serializable_dict, f, ensure_ascii=False, indent=4)
        
        