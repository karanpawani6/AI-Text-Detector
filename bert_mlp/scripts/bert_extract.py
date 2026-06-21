
# =============================
# 0. CONFIG
# =============================
MAGE_SPLIT   = "train"     
MAX_PER_CLASS = 17000      
MIN_WORDS    = 150          
BATCH_SIZE   = 32           
REDUCE_DIM   = 80           
RANDOM_SEED  = 42
BERT_MODEL   = "bert-base-uncased"   #

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets

# ── BERT ───
import torch
from transformers import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"Loading tokenizer & model: {BERT_MODEL} …")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
bert      = BertModel.from_pretrained(BERT_MODEL).to(device)
bert.eval()
print("Model ready.\n")

# =============================
# 1. LOAD MAGE DATASET
# =============================

print("Downloading MAGE dataset from Hugging Face (yaful/MAGE) …")

ds_all = load_dataset("yaful/MAGE")   


all_splits = [ds_all[split] for split in ds_all.keys()]
ds_combined = concatenate_datasets(all_splits)

print(f"MAGE total rows (all splits combined): {len(ds_combined)}")

df_mage = ds_combined.to_pandas()[["text", "label"]]
df_mage = df_mage.dropna(subset=["text"])
df_mage["text"]  = df_mage["text"].astype(str)
df_mage["label"] = df_mage["label"].astype(int)

print(f"After NA drop          : {df_mage.shape}")
print(f"Raw label distribution :\n{df_mage['label'].value_counts()}\n")

df_mage = df_mage[df_mage["text"].apply(lambda t: len(t.split()) >= MIN_WORDS)]
df_mage = df_mage.reset_index(drop=True)
print(f"After length filter (≥{MIN_WORDS} words): {df_mage.shape}")

human_df = df_mage[df_mage["label"] == 0]
ai_df    = df_mage[df_mage["label"] == 1]

if len(human_df) < MAX_PER_CLASS:
    raise ValueError(
        f"Not enough human rows after filtering: "
        f"need {MAX_PER_CLASS}, got {len(human_df)}"
    )
if len(ai_df) < MAX_PER_CLASS:
    raise ValueError(
        f"Not enough AI rows after filtering: "
        f"need {MAX_PER_CLASS}, got {len(ai_df)}"
    )

human_df = human_df.sample(n=MAX_PER_CLASS, random_state=RANDOM_SEED)
ai_df    = ai_df.sample(n=MAX_PER_CLASS,    random_state=RANDOM_SEED)

df = pd.concat([human_df, ai_df]).reset_index(drop=True)
print(f"\nAfter balancing  : {df.shape}  ({MAX_PER_CLASS} human + {MAX_PER_CLASS} AI)")

# =============================
# 2. SHUFFLE
# =============================
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Shuffled dataset : {df.shape}")
print(f"Labels           :\n{df['label'].value_counts()}\n")

# =============================
# 3. BERT EMBEDDING EXTRACTION
# =============================
def get_bert_embeddings(texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Returns mean-pooled token embeddings for each text.
    Shape: (len(texts), 768)

    Strategy: truncate to 512 tokens (BERT limit).
    Mean-pooling over non-padding tokens is more robust than CLS alone
    for variable-length inputs.
    """
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # ── Mean-pool over non-padding tokens ────────────────────────────────
        last_hidden = outputs.last_hidden_state          # (B, seq_len, 768)
        mask_exp    = attention_mask.unsqueeze(-1).float()
        sum_hidden  = (last_hidden * mask_exp).sum(dim=1)
        count       = mask_exp.sum(dim=1)
        mean_hidden = (sum_hidden / count).cpu().numpy()  # (B, 768)

        all_embeddings.append(mean_hidden)

        done = min(start + batch_size, len(texts))
        if done % 1000 < batch_size or done == len(texts):
            print(f"  Encoded {done:>6}/{len(texts)}", flush=True)

    return np.vstack(all_embeddings)   # (N, 768)


print("Extracting BERT embeddings …")
texts      = df["text"].tolist()
embeddings = get_bert_embeddings(texts)
print(f"Embeddings shape: {embeddings.shape}\n")

# =============================
# 4. OPTIONAL PCA REDUCTION
# =============================
if REDUCE_DIM is not None and REDUCE_DIM < embeddings.shape[1]:
    print(f"Applying PCA: 768 → {REDUCE_DIM} dimensions …")
    from sklearn.decomposition import PCA
    pca        = PCA(n_components=REDUCE_DIM, random_state=RANDOM_SEED)
    embeddings = pca.fit_transform(embeddings)
    explained  = pca.explained_variance_ratio_.sum() * 100
    print(f"Variance explained by {REDUCE_DIM} components: {explained:.2f}%\n")

    import joblib
    joblib.dump(pca, "bert_pca.pkl")
    print("PCA model saved → bert_pca.pkl")

final_dim = embeddings.shape[1]
bert_cols = [f"bert_{i}" for i in range(final_dim)]

# =============================
# 5. ASSEMBLE & SAVE
# =============================
emb_df   = pd.DataFrame(embeddings, columns=bert_cols)
final_df = pd.concat([df[["label"]].reset_index(drop=True), emb_df], axis=1)

output_path = "extracted_bert_combined.csv"
final_df.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}   shape={final_df.shape}")
print(f"Columns: label + {final_dim} BERT features\n")

# =============================
# 6. SANITY CHECK
# =============================
print("--- Sanity check: mean of first 5 BERT dims per class ---")
for col in bert_cols[:5]:
    h = final_df[final_df["label"] == 0][col].mean()
    a = final_df[final_df["label"] == 1][col].mean()
    print(f"  {col}   human={h:.4f}   ai={a:.4f}")

print("\nDone.")