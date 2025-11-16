import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from functools import partial
 
# ================= Configuration Parameters =================
# Please modify the following paths according to your actual directory structure
# Change the embedding file paths to relative paths or your actual paths
embedding_list = [
    './embeddings/embedding_CN_gemma-2-9b-it/',
    # './embeddings/embedding_CN_Llama-3.1-8B/',
    # './embeddings/embedding_CN_bert-base-uncased/',
    # './embeddings/embedding_CN_DeepSeek-R1-Distill-Qwen-7B/',
    # './embeddings/embedding_CN_Baichuan-7B/', 
    # './embeddings/embedding_CN_Baichuan2-7B-Chat/',
    # './embeddings/embedding_CN_Qwen2.5-7B/',
    # './embeddings/embedding_CN_opt-6.7b/',
    # './embeddings/embedding_CN_Llama-3.1-8B-Instruct/',
    # './embeddings/embedding_CN_gemma-2-9b/',
    # './embeddings/embedding_CN_Qwen2.5-7B-Instruct/',
    # './embeddings/embedding_CN_Mistral-7B-Instruct-v0.3/',
    # './embeddings/embedding_CN_Mistral-7B-v0.3/',
    # './embeddings/embedding_CN_glm-4-9b-chat-hf/',
    # './embeddings/embedding_CN_glm-4-9b-hf/',
]

# roi_config = """
# Frontal_Inf_Orb_L,15
# Frontal_Inf_Tri_L,13
# Frontal_Mid_L,7
# Temporal_Pole_Sup_L,83
# Temporal_Mid_L,85
# Angular_L,65
# Frontal_Inf_Orb_R,16
# Frontal_Inf_Tri_R,14
# Frontal_Mid_R,8
# Temporal_Pole_Sup_R,84
# Temporal_Mid_R,86
# Angular_R,66
# """
# roi_config ="""
# LH_IFGorb,0
# LH_IFG,1
# LH_MFG,2
# LH_AntTemp,3
# LH_PostTemp,4
# LH_AngG,5
# RH_IFGorb,6
# RH_IFG,7
# RH_MFG,8
# RH_AntTemp,9
# RH_PostTemp,10
# RH_AngG,11
# """
roi_config ="""
LH_IFGorb,0
"""
beta_path = "all_subjects_beta_CN_federenko_mask.npy"
n_subjects = 34  # Number of subjects
# ============================================

# Parse ROI configuration
ROI_info = []
for line in roi_config.strip().split('\n'):
    name, idx = line.split(',')
    ROI_info.append((name.strip(), int(idx.strip())))

# Define standardization function
def standardize_data(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

# Core processing function
def process_single_case(subj, embeddings, Y, roi_idx, model_name, roi_name):
    try:
        Y_subj = standardize_data(Y[subj])
        y = Y_subj[:, roi_idx]
        
        # Cross-validation
        def enhanced_xval(X, y):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, test_idx in kf.split(X):
                model = RidgeCV(alphas=np.logspace(-3, 1, 10))
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                scores.append(np.corrcoef(y[test_idx], pred)[0, 1])
            return np.mean(scores), np.std(scores)
        
        # Use partial function to encapsulate parameters
        def process_layer(layer):
            X_layer = embeddings[:, layer, :]  
            mean_corr, std_corr = enhanced_xval(X_layer, y)
            return layer, mean_corr, std_corr
        
        # Parallel execution
        layer_results = Parallel(n_jobs=-1)(
            delayed(process_layer)(layer) 
            for layer in range(n_layers)
        )
        
        # Process results
        layers, corrs, stds = zip(*sorted(layer_results))
        
        # Save individual results
        np.savez(
            f"results_language_new/{model_name}/subj_{subj}_{roi_name}.npz",
            correlations=corrs,
            std_devs=stds
        )
        
        # Generate individual charts
        plt.figure(figsize=(12, 6))
        plt.errorbar(layers, corrs, yerr=stds, fmt='-o', capsize=5)
        plt.title(f"{model_name} - Subj{subj} - {roi_name}")
        plt.savefig(f"figures_language_new/{model_name}/subj_{subj}_{roi_name}.png")
        plt.close()
        
        return corrs
    
    except Exception as e:
        print(f"Error processing case: {model_name}-{roi_name}-Subj{subj}: {str(e)}")
        return None

for model_path in embedding_list:
    # Get model name
    model_name = os.path.basename(model_path.rstrip('/')).split('_')[-1]
    
    # Create output directories
    os.makedirs(f"results_language_new/{model_name}", exist_ok=True)
    os.makedirs(f"figures_language_new/{model_name}", exist_ok=True)
    
    # Load embedding data
    embeddings = np.concatenate([
        np.load(f"{model_path}/{i}.npy").mean(axis=2) 
        for i in range(1577)
    ], axis=0)
    _, n_layers, _ = embeddings.shape
    # Load fMRI data
    Y = np.load(beta_path)  # (34, 1577, 90)
    
    # Iterate through all ROIs
    for roi_name, roi_idx in ROI_info:
        print(f"\n{'='*40}\nProcessing {model_name} - {roi_name}\n{'='*40}")
        
        # Process all subjects in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_single_case)(subj, embeddings, Y, roi_idx, model_name, roi_name)
            for subj in range(n_subjects)
        )
        
        # Calculate and save group average results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            avg_corr = np.mean(valid_results, axis=0)
            
            # Save average results
            np.savez(
                f"results_language_new/{model_name}/average_{roi_name}.npz",
                mean_correlation=avg_corr,
                valid_subjects=len(valid_results)
            )
            
            # Generate average charts
            plt.figure(figsize=(12, 6))
            plt.plot(avg_corr, '-o', linewidth=2)
            plt.title(f"{model_name} - Average - {roi_name}")
            plt.savefig(f"figures_language_new/{model_name}/average_{roi_name}.png")
            plt.close()

print("\nAll analysis completed!")

