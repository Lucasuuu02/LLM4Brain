# ================= Configuration Parameters =================
# Please modify the following paths according to your actual directory structure
# Change the embedding file paths to relative paths or your actual paths
embedding_list = [
    './embeddings/embedding_CN_gemma-2-9b-it_scheme2/',
    # './embeddings/embedding_CN_Llama-3.1-8B_scheme2/',
    # './embeddings/embedding_CN_bert-base-uncased_scheme2/',
    # './embeddings/embedding_CN_DeepSeek-R1-Distill-Qwen-7B_scheme2/',
    # './embeddings/embedding_CN_Baichuan-7B_scheme2/', 
    # './embeddings/embedding_CN_Baichuan2-7B-Chat_scheme2/',
    # './embeddings/embedding_CN_Qwen2.5-7B_scheme2/',
    # './embeddings/embedding_CN_opt-6.7b_scheme2/',
    # './embeddings/embedding_CN_Llama-3.1-8B-Instruct_scheme2/',
    # './embeddings/embedding_CN_gemma-2-9b_scheme2/',
    # './embeddings/embedding_CN_Qwen2.5-7B-Instruct_scheme2/',
    # './embeddings/embedding_CN_Mistral-7B-Instruct-v0.3_scheme2/',
    # './embeddings/embedding_CN_Mistral-7B-v0.3_scheme2/',
    # './embeddings/embedding_CN_glm-4-9b-chat-hf_scheme2/',
    # './embeddings/embedding_CN_glm-4-9b-hf_scheme2/',
]


# betas_df = pd.read_csv('test/final_betas.csv')
# betas = betas_df.values  # Convert concatenated betas_df to NumPy array (1577, ROI)
betas = df_combined
betas = betas.values 
# Store results
results = {}
alphas = np.logspace(-3, 5, 20)  # 20 values from [10^-3 to 10^5]

def ridge_closed_form(X_gpu, y_gpu, alpha):
    """Ridge regression closed-form solution: w = (XᵀX + αI)^(-1) Xᵀy"""
    n_features = X_gpu.shape[1]
    I_gpu = cp.identity(n_features)
    return cp.linalg.solve(X_gpu.T @ X_gpu + alpha * I_gpu, X_gpu.T @ y_gpu)


def standardized_data_on_gpu(X_train_np, X_test_np, y_train_np):
    """Standardize input and convert to GPU format"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_np)
    X_test = scaler.transform(X_test_np)

    return cp.asarray(X_train), cp.asarray(X_test), cp.asarray(y_train_np)


# Iterate through each model path
for model_path in embedding_list:
    model_name = os.path.basename(model_path.rstrip("/"))  # Extract model name
    print(f"Processing model: {model_name}")

    # Step 1: Load embeddings for all sentences, forming (1577, layers, size)
    embeddings = []
    for i in range(0, 1577):  # Iterate through files 0-1576
        npy_path = os.path.join(model_path, f"{i}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"File {npy_path} does not exist!")
        embedding = np.load(npy_path)  # Load .npy file
        embeddings.append(np.squeeze(embedding))  # Remove first dimension from (1, layers, size)

    embeddings = np.array(embeddings)  # Convert to NumPy array with shape (1577, layers, size)
    n_layers = embeddings.shape[1]  # Number of layers
    size = embeddings.shape[2]  # Feature dimension of each layer

    # Step 2: Build regression model for each layer and calculate correlations
    embeddings = np.array(embeddings)  # (1577, layers, size)
    n_layers = embeddings.shape[1]
    size = embeddings.shape[2]
    n_rois = betas.shape[1]  # Number of brain regions

    # Initialize model results dictionary
    model_results = {
    'correlations': np.zeros((n_layers, n_rois)),
    'p_values': np.zeros((n_layers, n_rois)),
    'best_alphas': np.zeros((n_layers, n_rois)) 
    }

    # Step 2: Iterate through each layer and each brain region, perform linear regression + calculate correlations
    print(f"Running regression for {n_layers} layers and {n_rois} ROIs...")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for layer in tqdm(range(5), desc="Layers"):
        X_cpu = embeddings[:, layer, :]
        
        for roi in range(n_rois):
            y_cpu = betas[:, roi]

            y_pred_all = []
            y_true_all = []
            best_alpha_list = []
            corrs_per_fold = []

            for train_idx, test_idx in kf.split(X_cpu):
                # Extract current fold
                X_train_np = X_cpu[train_idx]
                X_test_np = X_cpu[test_idx]
                y_train_np = y_cpu[train_idx]
                y_test_np = y_cpu[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_np)
                X_test = scaler.transform(X_test_np)

                # Convert to GPU
                X_train_gpu = cp.asarray(X_train)
                X_test_gpu = cp.asarray(X_test)
                y_train_gpu = cp.asarray(y_train_np)

                best_score = -np.inf
                best_alpha = None
                best_pred = None

                for alpha in alphas:
                    # Train
                    w = ridge_closed_form(X_train_gpu, y_train_gpu, alpha)

                    # Predict
                    y_pred_gpu = X_test_gpu @ w
                    y_pred_np = cp.asnumpy(y_pred_gpu)

                    # Evaluate (on CPU)
                    corr, _ = pearsonr(y_test_np, y_pred_np)
                    if corr > best_score:
                        best_score = corr
                        best_alpha = alpha
                        best_pred = y_pred_np

                y_pred_all.extend(best_pred)
                y_true_all.extend(y_test_np)
                best_alpha_list.append(best_alpha)

                corr_this_fold, _ = pearsonr(y_test_np, best_pred)
                corrs_per_fold.append(corr_this_fold)


            print(f"[Layer {layer}, ROI {roi}] fold corrs: {corrs_per_fold}, mean: {np.mean(corrs_per_fold):.3f}")

            final_corr, final_pval = pearsonr(y_true_all, y_pred_all)
            model_results['correlations'][layer, roi] = final_corr
            model_results['p_values'][layer, roi] = final_pval
            model_results['best_alphas'][layer, roi] = np.mean(best_alpha_list)

    results[model_name] = model_results


for model_name, model_result in results.items():
    corr_df = pd.DataFrame(model_result['correlations'], columns=[f'ROI_{i}' for i in range(n_rois)])
    pval_df = pd.DataFrame(model_result['p_values'], columns=[f'ROI_{i}' for i in range(n_rois)])
    alpha_df = pd.DataFrame(model_result['best_alphas'], columns=[f'ROI_{i}' for i in range(n_rois)])

    corr_df.to_csv(f"{model_name}_ridge_correlations.csv", index_label='Layer')
    pval_df.to_csv(f"{model_name}_ridge_p_values.csv", index_label='Layer')
    alpha_df.to_csv(f"{model_name}_ridge_alphas.csv", index_label='Layer')

    print(f"Saved ridge regression results for {model_name}")