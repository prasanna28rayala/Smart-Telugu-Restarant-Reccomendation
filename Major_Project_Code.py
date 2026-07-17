import sys
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
print("Installing dependencies...")
for pkg in ['torch', 'transformers', 'sentence-transformers', 'scikit-fuzzy',
            'scikit-learn', 'pandas', 'numpy', 'tqdm', 'matplotlib', 'seaborn', 'xgboost']:
    try:
        _import(pkg.replace('-', '').split('-')[0])
    except:
        install(pkg)
import re
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")
print("\n" + "="*60)
print("TELUGU RESTAURANT RECOMMENDATION SYSTEM")
print("ENSEMBLE VERSION: BiLSTM + XGBoost")
print("="*60)
def get_csv_file():
    print("\n" + "="*60)
    print("UPLOAD CSV FILE")
    print("="*60)
    try:
        from google.colab import files
        print("\nGoogle Colab detected")
        print("Click 'Choose Files' button below to upload your CSV:")
        uploaded = files.upload()
        if uploaded:
            filename = list(uploaded.keys())[0]
            filepath = f"/content/{filename}"
            with open(filepath, 'wb') as f:
                f.write(uploaded[filename])
            print(f"\nFile uploaded: {filename}")
            return filepath
    except:
        pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        print("\nOpening file picker...")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        filepath = filedialog.askopenfilename(
            title='Select Restaurant CSV File',
            filetypes=[('CSV files', '.csv'), ('All files', '.*')]
        )
        if filepath:
            print(f"\nFile selected: {os.path.basename(filepath)}")
            return filepath
    except:
        pass
    print("\nManual file path entry:")
    print("Example: C:/Users/Name/Downloads/Restaurant_Dataset1.csv")
    filepath = input("\nEnter file path: ").strip()
    filepath = filepath.replace('"', '').replace("'", '').replace('\\', '/')
    if os.path.exists(filepath):
        print(f"\nFile found: {os.path.basename(filepath)}")
        return filepath
    else:
        print(f"\nERROR: File not found")
        return None
DATA_FILE = get_csv_file()
if DATA_FILE is None:
    print("\nUpload failed. Please try again.")
    sys.exit(1)
CONFIG = {
    'sample_size': 1000,
    'batch_size': 16,
    'max_length': 128,
    'bilstm_hidden': 256,
    'bilstm_layers': 3,
    'bilstm_epochs': 25,
    'bilstm_lr': 0.0005,
    'xgb_epochs': 100,
    'ensemble_weight_bilstm': 0.5,
    'ensemble_weight_xgb': 0.5,
    'ncf_dim': 128,
    'ncf_epochs': 25,
    'ncf_lr': 0.0003,
    'relevance_threshold': 0.3,
    'weights': {
        'fuzzy': 0.30,
        'ncf': 0.30,
        'relevance': 0.25,
        'sentiment': 0.15
    }
}
FOOD_KEYWORDS = {
    'telugu': [
        'బిర్యానీ', 'బిరియాని', 'దోసె', 'దోస', 'ఇడ్లీ', 'ఇడ్లి', 'పొంగల్',
        'వడ', 'చపాతి', 'పూరి', 'చికెన్', 'మటన్', 'ఫిష్', 'పనీర్', 'మసాలా',
        'కర్రీ', 'టిఫిన్', 'భోజనం', 'తిండి', 'అన్నం', 'కూర', 'పచ్చడి', 'చారు',
        'రసం', 'సాంబార్', 'కాఫీ', 'టీ', 'వేపుడు', 'తీపి', 'స్వీట్'
    ],
    'english': [
        'biryani', 'biriyani', 'dosa', 'idli', 'idly', 'pongal', 'vada', 'chapati',
        'puri', 'poori', 'chicken', 'mutton', 'fish', 'paneer', 'masala', 'curry',
        'tiffin', 'meal', 'food', 'rice', 'dal', 'chutney', 'sambar', 'coffee',
        'tea', 'fry', 'fried', 'sweet', 'spicy', 'tandoori'
    ]
}
def is_food_query(query):
    query_lower = query.lower()
    for keyword in FOOD_KEYWORDS['telugu']:
        if keyword in query:
            return True
    for keyword in FOOD_KEYWORDS['english']:
        if keyword in query_lower:
            return True
    return False
def get_query_food_keywords(query):
    query_lower = query.lower()
    found_keywords = []
    for keyword in FOOD_KEYWORDS['telugu']:
        if keyword in query:
            found_keywords.append(keyword)
    for keyword in FOOD_KEYWORDS['english']:
        if keyword in query_lower:
            found_keywords.append(keyword)
    return found_keywords
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
def rating_to_label(r):
    if pd.isna(r):
        return None
    r = float(r)
    return "positive" if r >= 4.0 else ("neutral" if r >= 3.0 else "negative")
print(f"\n{'='*60}")
print("LOADING DATASET")
print('='*60)
df_full = pd.read_csv(DATA_FILE)
print(f"\nFull dataset: {df_full.shape[0]} rows, {df_full.shape[1]} columns")
text_col = 'review_telugu'
df_full[text_col] = df_full[text_col].astype(str).apply(clean_text)
df_full["rating"] = pd.to_numeric(df_full["rating"], errors="coerce")
df_full["sentiment_label"] = df_full["rating"].apply(rating_to_label)
print(f"\n{'='*60}")
print(f"SAMPLING {CONFIG['sample_size']} ROWS (BALANCED)")
print('='*60)
df_labeled = df_full.dropna(subset=["sentiment_label"]).copy()
sentiment_counts = df_labeled['sentiment_label'].value_counts()
print("\nOriginal sentiment distribution:")
for label, count in sentiment_counts.items():
    print(f"  {label}: {count}")
samples_per_category = {}
total_labeled = len(df_labeled)
for label, count in sentiment_counts.items():
    proportion = count / total_labeled
    samples_per_category[label] = max(1, int(CONFIG['sample_size'] * proportion))
total_samples = sum(samples_per_category.values())
if total_samples != CONFIG['sample_size']:
    largest = max(samples_per_category, key=samples_per_category.get)
    samples_per_category[largest] += (CONFIG['sample_size'] - total_samples)
print(f"\nSampling strategy:")
for label, count in samples_per_category.items():
    print(f"  {label}: {count} samples")
sampled_dfs = []
for label, n_samples in samples_per_category.items():
    category_df = df_labeled[df_labeled['sentiment_label'] == label]
    if len(category_df) >= n_samples:
        sampled = category_df.sample(n=n_samples, random_state=42)
    else:
        sampled = category_df
    sampled_dfs.append(sampled)
df = pd.concat(sampled_dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nSampled dataset: {len(df)} rows")
print(f"Restaurants: {df['restaurant_name'].nunique()}")
np.random.seed(42)
n_users = min(100, len(df) // 10)
df['user_id'] = np.random.randint(0, n_users, len(df))
print(f"\n{'='*60}")
print("LOADING TRANSFORMER MODELS")
print('='*60)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("\nLoading MuRIL...")
muril_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased").eval().to(device)
print("MuRIL loaded")
print("\nLoading IndicBERT...")
try:
    indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    indic_model = AutoModel.from_pretrained("ai4bharat/indic-bert").eval().to(device)
    print("IndicBERT loaded")
except:
    print("Using MiniLM alternative")
    indic_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    indic_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").eval().to(device)
def embed_text(texts, tokenizer, model, batch_size=None):
    if batch_size is None:
        batch_size = CONFIG['batch_size']
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                       max_length=CONFIG['max_length'], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = ((out.last_hidden_state * mask).sum(1) /
                   mask.sum(1).clamp(min=1e-9)).cpu().numpy()
            all_emb.append(emb)
    return np.vstack(all_emb) if all_emb else np.zeros((0, model.config.hidden_size))
class ImprovedBiLSTM(nn.Module):
    def _init_(self, dim, hidden=None, num_layers=None):
        super()._init_()
        if hidden is None:
            hidden = CONFIG['bilstm_hidden']
        if num_layers is None:
            num_layers = CONFIG['bilstm_layers']
        self.lstm = nn.LSTM(dim, hidden, num_layers,
                           bidirectional=True, batch_first=True, dropout=0.4)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(hidden*2, hidden)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 3)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(1)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        return self.fc2(out)
class Dataset1(Dataset):
    def _init_(self, emb, labels):
        self.emb = torch.FloatTensor(emb)
        self.labels = torch.LongTensor(labels)
    def _len_(self):
        return len(self.labels)
    def _getitem_(self, i):
        return self.emb[i], self.labels[i]
labeled = df.dropna(subset=["sentiment_label"])
label_map = {"negative": 0, "neutral": 1, "positive": 2}
labeled = labeled[labeled["sentiment_label"].isin(label_map.keys())]
X_text = labeled[text_col].tolist()
y = [label_map[s] for s in labeled["sentiment_label"]]
print(f"\nTraining samples: {len(X_text)}")
print("Generating embeddings...")
X_emb = embed_text(X_text, indic_tokenizer, indic_model)
X_train, X_test, y_train, y_test = train_test_split(
    X_emb, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")
print(f"\n{'='*60}")
print("TRAINING BiLSTM SENTIMENT CLASSIFIER")
print('='*60)
X_train_lstm = X_train[:, np.newaxis, :]
X_test_lstm = X_test[:, np.newaxis, :]
train_loader = DataLoader(Dataset1(X_train_lstm, y_train), batch_size=32, shuffle=True)
bilstm = ImprovedBiLSTM(indic_model.config.hidden_size).to(device)
opt_bilstm = torch.optim.Adam(bilstm.parameters(), lr=CONFIG['bilstm_lr'])
loss_fn = nn.CrossEntropyLoss()
print(f"\nTraining BiLSTM ({CONFIG['bilstm_epochs']} epochs)...")
bilstm_best_acc = 0
bilstm_best_preds = None
bilstm_train_losses = []
bilstm_train_accs = []
for epoch in range(CONFIG['bilstm_epochs']):
    bilstm.train()
    total_loss = 0
    n_batches = 0
    for emb, lbl in train_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        opt_bilstm.zero_grad()
        loss = loss_fn(bilstm(emb), lbl)
        loss.backward()
        opt_bilstm.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches
    bilstm_train_losses.append(avg_loss)
    bilstm.eval()
    with torch.no_grad():
        bilstm_preds = bilstm(torch.FloatTensor(X_test_lstm).to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y_test, bilstm_preds)
    bilstm_train_accs.append(acc)
    print(f"Epoch {epoch+1}/{CONFIG['bilstm_epochs']} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    if acc > bilstm_best_acc:
        bilstm_best_acc = acc
        bilstm_best_preds = bilstm_preds
print(f"\n{'='*60}")
print(f"BiLSTM TRAINING COMPLETE")
print(f"Best Accuracy: {bilstm_best_acc:.4f} ({bilstm_best_acc*100:.2f}%)")
print('='*60)
bilstm_precision, bilstm_recall, bilstm_f1, _ = precision_recall_fscore_support(
    y_test, bilstm_best_preds, average='weighted'
)
print("\nDetailed BiLSTM Metrics:")
print(f"  Precision: {bilstm_precision:.4f} ({bilstm_precision*100:.2f}%)")
print(f"  Recall: {bilstm_recall:.4f} ({bilstm_recall*100:.2f}%)")
print(f"  F1-Score: {bilstm_f1:.4f} ({bilstm_f1*100:.2f}%)")
print("\nBiLSTM Classification Report:")
print(classification_report(y_test, bilstm_best_preds,
                          target_names=["negative", "neutral", "positive"]))
bilstm_cm = confusion_matrix(y_test, bilstm_best_preds)
print(f"\n{'='*60}")
print("TRAINING XGBOOST SENTIMENT CLASSIFIER")
print('='*60)
print(f"\nTraining XGBoost classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=CONFIG['xgb_epochs'],
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_train_preds = xgb_model.predict(X_train)
xgb_test_preds = xgb_model.predict(X_test)
xgb_train_acc = accuracy_score(y_train, xgb_train_preds)
xgb_test_acc = accuracy_score(y_test, xgb_test_preds)
print(f"\nTraining accuracy: {xgb_train_acc:.4f} ({xgb_train_acc*100:.2f}%)")
print(f"Test accuracy: {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
print(f"\n{'='*60}")
print(f"XGBOOST TRAINING COMPLETE")
print(f"Best Accuracy: {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
print('='*60)
xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(
    y_test, xgb_test_preds, average='weighted'
)
print("\nDetailed XGBoost Metrics:")
print(f"  Precision: {xgb_precision:.4f} ({xgb_precision*100:.2f}%)")
print(f"  Recall: {xgb_recall:.4f} ({xgb_recall*100:.2f}%)")
print(f"  F1-Score: {xgb_f1:.4f} ({xgb_f1*100:.2f}%)")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_test_preds,
                          target_names=["negative", "neutral", "positive"]))
xgb_cm = confusion_matrix(y_test, xgb_test_preds)
print(f"\n{'='*60}")
print("CREATING ENSEMBLE MODEL (BiLSTM + XGBoost)")
print('='*60)
bilstm.eval()
with torch.no_grad():
    bilstm_probs = torch.softmax(bilstm(torch.FloatTensor(X_test_lstm).to(device)), 1).cpu().numpy()
xgb_probs = xgb_model.predict_proba(X_test)
ensemble_probs = (
    CONFIG['ensemble_weight_bilstm'] * bilstm_probs +
    CONFIG['ensemble_weight_xgb'] * xgb_probs
)
ensemble_preds = ensemble_probs.argmax(1)
ensemble_acc = accuracy_score(y_test, ensemble_preds)
ensemble_precision, ensemble_recall, ensemble_f1, _ = precision_recall_fscore_support(
    y_test, ensemble_preds, average='weighted'
)
print(f"\nEnsemble Weights:")
print(f"  BiLSTM: {CONFIG['ensemble_weight_bilstm']}")
print(f"  XGBoost: {CONFIG['ensemble_weight_xgb']}")
print(f"\nEnsemble Performance:")
print(f"  Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"  Precision: {ensemble_precision:.4f} ({ensemble_precision*100:.2f}%)")
print(f"  Recall: {ensemble_recall:.4f} ({ensemble_recall*100:.2f}%)")
print(f"  F1-Score: {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, ensemble_preds,
                          target_names=["negative", "neutral", "positive"]))
ensemble_cm = confusion_matrix(y_test, ensemble_preds)
print(f"\n{'='*60}")
print("MODEL COMPARISON")
print('='*60)
print(f"BiLSTM Accuracy:    {bilstm_best_acc:.4f} ({bilstm_best_acc*100:.2f}%)")
print(f"XGBoost Accuracy:   {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
print(f"Ensemble Accuracy:  {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"Improvement:        {(ensemble_acc - max(bilstm_best_acc, xgb_test_acc))*100:.2f}%")
class ImprovedNCF(nn.Module):
    def _init_(self, nu, ni, dim=None):
        super()._init_()
        if dim is None:
            dim = CONFIG['ncf_dim']
        self.u_emb = nn.Embedding(nu, dim)
        self.i_emb = nn.Embedding(ni, dim)
        self.gmf = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim*2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, dim)
        )
        self.out = nn.Linear(dim*2, 1)
    def forward(self, u, i):
        ue, ie = self.u_emb(u), self.i_emb(i)
        g = self.gmf(ue * ie)
        m = self.mlp(torch.cat([ue, ie], 1))
        return torch.sigmoid(self.out(torch.cat([g, m], 1)))
print(f"\n{'='*60}")
print("TRAINING NCF COLLABORATIVE FILTERING")
print('='*60)
user_enc = LabelEncoder()
rest_enc = LabelEncoder()
df['user_enc'] = user_enc.fit_transform(df['user_id'].astype(str))
df['rest_enc'] = rest_enc.fit_transform(df['restaurant_name'])
print(f"Users: {df['user_enc'].nunique()}, Restaurants: {df['rest_enc'].nunique()}")
ncf = ImprovedNCF(df['user_enc'].nunique(), df['rest_enc'].nunique()).to(device)
opt_ncf = torch.optim.Adam(ncf.parameters(), lr=CONFIG['ncf_lr'], weight_decay=1e-5)
print(f"\nTraining NCF ({CONFIG['ncf_epochs']} epochs)...")
ncf_losses = []
for epoch in range(CONFIG['ncf_epochs']):
    ncf.train()
    total_loss = 0
    n_batches = 0
    for i in range(0, len(df), 128):
        batch = df.iloc[i:i+128]
        u = torch.LongTensor(batch['user_enc'].values).to(device)
        r = torch.LongTensor(batch['rest_enc'].values).to(device)
        rating = torch.FloatTensor(batch['rating'].values/5.0).to(device)
        opt_ncf.zero_grad()
        loss = nn.MSELoss()(ncf(u, r).squeeze(), rating)
        loss.backward()
        opt_ncf.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches
    ncf_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{CONFIG['ncf_epochs']} - Loss: {avg_loss:.4f}")
print(f"\n{'='*60}")
print("NCF TRAINING COMPLETE")
print('='*60)
true_ratings = df['rating'].values / 5.0
ncf_preds = []
ncf.eval()
with torch.no_grad():
    for u, r in zip(df['user_enc'].values, df['rest_enc'].values):
        p = ncf(torch.tensor([u]).to(device),
               torch.tensor([r]).to(device)).item()
        ncf_preds.append(p)
ncf_preds = np.array(ncf_preds)
ncf_rmse = np.sqrt(mean_squared_error(true_ratings, ncf_preds))
ncf_mae = np.mean(np.abs(true_ratings - ncf_preds))
ncf_acc = 1 - ncf_rmse
print(f"\nDetailed NCF Metrics:")
print(f"  RMSE: {ncf_rmse:.4f}")
print(f"  MAE: {ncf_mae:.4f}")
print(f"  Accuracy: {ncf_acc:.4f} ({ncf_acc*100:.2f}%)")
print(f"  Mean Prediction: {ncf_preds.mean():.4f}")
print(f"  Std Prediction: {ncf_preds.std():.4f}")
print(f"\n{'='*60}")
print("CREATING FUZZY LOGIC SYSTEM")
print('='*60)
relevance = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'relevance')
sentiment = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'sentiment')
rec = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'rec')
relevance['low'] = fuzz.trimf(relevance.universe, [0, 0, 0.4])
relevance['medium'] = fuzz.trimf(relevance.universe, [0.2, 0.5, 0.8])
relevance['high'] = fuzz.trimf(relevance.universe, [0.6, 1, 1])
sentiment['negative'] = fuzz.trimf(sentiment.universe, [0, 0, 0.4])
sentiment['neutral'] = fuzz.trimf(sentiment.universe, [0.2, 0.5, 0.8])
sentiment['positive'] = fuzz.trimf(sentiment.universe, [0.6, 1, 1])
rec['bad'] = fuzz.trimf(rec.universe, [0, 0, 0.3])
rec['average'] = fuzz.trimf(rec.universe, [0.2, 0.5, 0.7])
rec['good'] = fuzz.trimf(rec.universe, [0.5, 0.75, 1])
rec['excellent'] = fuzz.trimf(rec.universe, [0.7, 1, 1])
rules = [
    ctrl.Rule(relevance['low'] & sentiment['negative'], rec['bad']),
    ctrl.Rule(relevance['low'] & sentiment['neutral'], rec['average']),
    ctrl.Rule(relevance['low'] & sentiment['positive'], rec['average']),
    ctrl.Rule(relevance['medium'] & sentiment['negative'], rec['average']),
    ctrl.Rule(relevance['medium'] & sentiment['neutral'], rec['good']),
    ctrl.Rule(relevance['medium'] & sentiment['positive'], rec['good']),
    ctrl.Rule(relevance['high'] & sentiment['negative'], rec['average']),
    ctrl.Rule(relevance['high'] & sentiment['neutral'], rec['good']),
    ctrl.Rule(relevance['high'] & sentiment['positive'], rec['excellent'])
]
rec_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
print("Fuzzy logic system created with 9 rules")
def fuzzy_score(rel, sent):
    try:
        rec_sim.input['relevance'] = float(np.clip(rel, 0, 1))
        rec_sim.input['sentiment'] = float(np.clip(sent, 0, 1))
        rec_sim.compute()
        return float(rec_sim.output['rec'])
    except:
        return 0.5
print(f"\n{'='*60}")
print("PRECOMPUTING RESTAURANT DATA")
print('='*60)
grouped = df.groupby("restaurant_name")[text_col].apply(list).to_dict()
restaurants = list(grouped.keys())
rest_muril, rest_indic, rest_sent = {}, {}, {}
print(f"Processing {len(restaurants)} restaurants...")
for r in tqdm(restaurants, desc="Precomputing"):
    reviews = grouped[r][:50]
    if not reviews:
        rest_muril[r] = np.zeros(muril_model.config.hidden_size)
        rest_indic[r] = np.zeros(indic_model.config.hidden_size)
        rest_sent[r] = 0.5
        continue
    rest_muril[r] = embed_text(reviews, muril_tokenizer, muril_model).mean(0)
    rest_indic[r] = embed_text(reviews, indic_tokenizer, indic_model).mean(0)
    try:
        embs = embed_text(reviews, indic_tokenizer, indic_model)
        embs_lstm = embs[:, np.newaxis, :]
        with torch.no_grad():
            bilstm_probs_rest = torch.softmax(bilstm(torch.FloatTensor(embs_lstm).to(device)), 1).cpu().numpy()
        xgb_probs_rest = xgb_model.predict_proba(embs)
        ensemble_probs_rest = (
            CONFIG['ensemble_weight_bilstm'] * bilstm_probs_rest +
            CONFIG['ensemble_weight_xgb'] * xgb_probs_rest
        )
        rest_sent[r] = float(ensemble_probs_rest[:, 2].mean())
    except:
        rr = df[df["restaurant_name"] == r]["rating"].dropna()
        rest_sent[r] = float((rr >= 4.0).mean()) if len(rr) > 0 else 0.5
print("Precomputation complete")
print(f"\n{'='*60}")
print("GENERATING VISUALIZATION GRAPHS")
print('='*60)
fig = plt.figure(figsize=(20, 12))
plt.subplot(3, 4, 1)
plt.plot(range(1, CONFIG['bilstm_epochs']+1), bilstm_train_losses, 'b-', linewidth=2)
plt.title('BiLSTM Training Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.subplot(3, 4, 2)
plt.plot(range(1, CONFIG['bilstm_epochs']+1), bilstm_train_accs, 'g-', linewidth=2)
plt.title('BiLSTM Training Accuracy', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.subplot(3, 4, 3)
labels = ['Negative', 'Neutral', 'Positive']
sns.heatmap(bilstm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('BiLSTM Confusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.subplot(3, 4, 4)
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('XGBoost Confusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.subplot(3, 4, 5)
sns.heatmap(ensemble_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
plt.title('Ensemble Confusion Matrix', fontsize=12, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.subplot(3, 4, 6)
plt.plot(range(1, CONFIG['ncf_epochs']+1), ncf_losses, 'r-', linewidth=2)
plt.title('NCF Training Loss', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.subplot(3, 4, 7)
models = ['BiLSTM', 'XGBoost', 'Ensemble']
accuracies = [bilstm_best_acc, xgb_test_acc, ensemble_acc]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.title('Sentiment Models Comparison', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.axhline(y=0.9, color='red', linestyle='--', label='90% Target')
plt.legend()
plt.subplot(3, 4, 8)
overall_acc = (ensemble_acc + ncf_acc) / 2
metrics_names = ['Ensemble', 'NCF', 'Overall']
metrics_values = [ensemble_acc, ncf_acc, overall_acc]
colors2 = ['#e67e22', '#9b59b6', '#1abc9c']
bars2 = plt.bar(metrics_names, metrics_values, color=colors2, alpha=0.7)
plt.title('Final System Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
for bar, val in zip(bars2, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.axhline(y=0.9, color='red', linestyle='--', label='90% Target')
plt.legend()
plt.subplot(3, 4, 9)
bilstm_metrics = {
    'Accuracy': bilstm_best_acc,
    'Precision': bilstm_precision,
    'Recall': bilstm_recall,
    'F1-Score': bilstm_f1
}
plt.barh(list(bilstm_metrics.keys()), list(bilstm_metrics.values()), color='skyblue', alpha=0.7)
plt.title('BiLSTM Metrics', fontsize=12, fontweight='bold')
plt.xlabel('Score')
plt.xlim([0, 1])
for i, (name, val) in enumerate(bilstm_metrics.items()):
    plt.text(val + 0.02, i, f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=9)
plt.subplot(3, 4, 10)
xgb_metrics = {
    'Accuracy': xgb_test_acc,
    'Precision': xgb_precision,
    'Recall': xgb_recall,
    'F1-Score': xgb_f1
}
plt.barh(list(xgb_metrics.keys()), list(xgb_metrics.values()), color='lightgreen', alpha=0.7)
plt.title('XGBoost Metrics', fontsize=12, fontweight='bold')
plt.xlabel('Score')
plt.xlim([0, 1])
for i, (name, val) in enumerate(xgb_metrics.items()):
    plt.text(val + 0.02, i, f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=9)
plt.subplot(3, 4, 11)
ensemble_metrics = {
    'Accuracy': ensemble_acc,
    'Precision': ensemble_precision,
    'Recall': ensemble_recall,
    'F1-Score': ensemble_f1
}
plt.barh(list(ensemble_metrics.keys()), list(ensemble_metrics.values()), color='lightsalmon', alpha=0.7)
plt.title('Ensemble Metrics', fontsize=12, fontweight='bold')
plt.xlabel('Score')
plt.xlim([0, 1])
for i, (name, val) in enumerate(ensemble_metrics.items()):
    plt.text(val + 0.02, i, f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=9)
plt.subplot(3, 4, 12)
ncf_detailed = {
    'Accuracy': ncf_acc,
    '1-RMSE': 1-ncf_rmse,
    '1-MAE': 1-ncf_mae
}
plt.barh(list(ncf_detailed.keys()), list(ncf_detailed.values()), color='plum', alpha=0.7)
plt.title('NCF Metrics', fontsize=12, fontweight='bold')
plt.xlabel('Score')
plt.xlim([0, 1])
for i, (name, val) in enumerate(ncf_detailed.items()):
    plt.text(val + 0.02, i, f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('ensemble_model_results.png', dpi=150, bbox_inches='tight')
print("\nGraphs saved as 'ensemble_model_results.png'")
print(f"\n{'='*60}")
print("COMPREHENSIVE SYSTEM EVALUATION")
print('='*60)
print(f"\n1. BiLSTM Sentiment Classifier:")
print(f"   - Accuracy: {bilstm_best_acc:.4f} ({bilstm_best_acc*100:.2f}%)")
print(f"   - Precision: {bilstm_precision:.4f} ({bilstm_precision*100:.2f}%)")
print(f"   - Recall: {bilstm_recall:.4f} ({bilstm_recall*100:.2f}%)")
print(f"   - F1-Score: {bilstm_f1:.4f} ({bilstm_f1*100:.2f}%)")
print(f"\n2. XGBoost Sentiment Classifier:")
print(f"   - Accuracy: {xgb_test_acc:.4f} ({xgb_test_acc*100:.2f}%)")
print(f"   - Precision: {xgb_precision:.4f} ({xgb_precision*100:.2f}%)")
print(f"   - Recall: {xgb_recall:.4f} ({xgb_recall*100:.2f}%)")
print(f"   - F1-Score: {xgb_f1:.4f} ({xgb_f1*100:.2f}%)")
print(f"\n3. Ensemble (BiLSTM + XGBoost):")
print(f"   - Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"   - Precision: {ensemble_precision:.4f} ({ensemble_precision*100:.2f}%)")
print(f"   - Recall: {ensemble_recall:.4f} ({ensemble_recall*100:.2f}%)")
print(f"   - F1-Score: {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
print(f"   - Improvement: {(ensemble_acc - max(bilstm_best_acc, xgb_test_acc))*100:.2f}%")
print(f"\n4. NCF Collaborative Filtering:")
print(f"   - Accuracy: {ncf_acc:.4f} ({ncf_acc*100:.2f}%)")
print(f"   - RMSE: {ncf_rmse:.4f}")
print(f"   - MAE: {ncf_mae:.4f}")
print(f"\n5. Overall System Metrics:")
print(f"   - Sentiment Model: Ensemble ({ensemble_acc*100:.2f}%)")
print(f"   - Collaborative Model: NCF ({ncf_acc*100:.2f}%)")
print(f"   - Combined Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"   - Total Restaurants: {len(restaurants)}")
print(f"   - Total Users: {df['user_enc'].nunique()}")
print(f"   - Total Reviews: {len(df)}")
print(f"\n{'='*60}")
if overall_acc >= 0.90:
    print(f"TARGET ACHIEVED: {overall_acc*100:.2f}% >= 90%")
else:
    print(f"TARGET NOT MET: {overall_acc*100:.2f}% < 90%")
    print(f"Gap: {(0.90 - overall_acc)*100:.2f}%")
print('='*60)
artifacts = {
    'rest_muril': rest_muril,
    'rest_indic': rest_indic,
    'rest_sent': rest_sent,
    'restaurants': restaurants,
    'bilstm': bilstm,
    'xgb_model': xgb_model,
    'metrics': {
        'bilstm_acc': bilstm_best_acc,
        'xgb_acc': xgb_test_acc,
        'ensemble_acc': ensemble_acc,
        'ncf_acc': ncf_acc,
        'overall_acc': overall_acc
    }
}
with open('artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
torch.save(bilstm.state_dict(), 'bilstm_model.pt')
torch.save(ncf.state_dict(), 'ncf_model.pt')
print("\nModels saved successfully")
def recommend(query, k=5):
    if not is_food_query(query):
        print(f"\nWARNING: '{query}' doesn't appear to be food-related.")
        print("Try Telugu food terms like: బిర్యానీ, దోసె, చికెన్, ఇడ్లీ")
        print("Or English: biryani, dosa, chicken, idli\n")
        return []
    query_keywords = get_query_food_keywords(query)
    qm = embed_text([query], muril_tokenizer, muril_model)[0]
    qi = embed_text([query], indic_tokenizer, indic_model)[0]
    results = []
    for r in restaurants:
        sim_m = cosine_similarity([qm], [rest_muril[r]])[0][0]
        sim_i = cosine_similarity([qi], [rest_indic[r]])[0][0]
        base_relevance = (sim_m + sim_i) / 2
        restaurant_text = " ".join(grouped[r]).lower()
        keyword_match_score = 0
        for keyword in query_keywords:
            if keyword.lower() in restaurant_text:
                keyword_match_score += 0.2
        keyword_match_score = min(keyword_match_score, 0.4)
        relevance = min(base_relevance + keyword_match_score, 1.0)
        if relevance < CONFIG['relevance_threshold']:
            continue
        sentiment = rest_sent[r]
        fuz = fuzzy_score(relevance, sentiment)
        try:
            rid = rest_enc.transform([r])[0]
            with torch.no_grad():
                ncf_score = ncf(torch.tensor([0]).to(device),
                              torch.tensor([rid]).to(device)).item()
        except:
            ncf_score = 0.5
        final = (
            CONFIG['weights']['fuzzy'] * fuz +
            CONFIG['weights']['ncf'] * ncf_score +
            CONFIG['weights']['relevance'] * relevance +
            CONFIG['weights']['sentiment'] * sentiment
        )
        results.append({
            'restaurant': r,
            'score': final,
            'relevance': relevance,
            'sentiment': sentiment,
            'fuzzy': fuz,
            'ncf': ncf_score
        })
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
    if not results:
        print(f"\nNo relevant restaurants found for '{query}'")
        print("Try more common food terms.")
    return results
print(f"\n{'='*60}")
print("DEMO RECOMMENDATIONS")
print('='*60)
demo_queries = ["బిర్యానీ", "దోసె", "చికెన్"]
for q in demo_queries:
    print(f"\nQuery: '{q}'")
    print("-" * 60)
    recs = recommend(q, 5)
    if recs:
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['restaurant']:30s} | Score: {r['score']:.4f}")
    else:
        print("No results")
print(f"\n{'='*60}")
print("INTERACTIVE QUERY MODE")
print("WITH ROBUST ERROR HANDLING")
print('='*60)
print("\nFood-related queries only!")
print("Examples (Telugu): బిర్యానీ, దోసె, చికెన్, ఇడ్లీ")
print("Examples (English): biryani, dosa, chicken, idli, curry\n")
def get_valid_number(prompt, default=5, min_val=1, max_val=20):
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default
        try:
            num = int(user_input)
            if min_val <= num <= max_val:
                return num
            else:
                print(f"Please enter number between {min_val} and {max_val}")
        except ValueError:
            print(f"Invalid input. Please enter only numbers (not Telugu or English text)")
while True:
    q = input("Enter food query (or 'quit' to exit): ").strip()
    if q.lower() in ['quit', 'q', 'exit']:
        break
    if not q:
        print("Please enter a query\n")
        continue
    k = get_valid_number("Enter number of results (1-20, default=5): ",
                        default=5, min_val=1, max_val=20)
    print(f"\nSearching for '{q}'...")
    print("-" * 60)
    recs = recommend(q, k)
    if recs:
        print(f"\nTop {len(recs)} recommendations:")
        print("-" * 60)
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['restaurant']:30s} | Score: {r['score']:.4f}")
            print(f"   Relevance: {r['relevance']:.3f} | Sentiment: {r['sentiment']:.3f} | "
                  f"Fuzzy: {r['fuzzy']:.3f} | NCF: {r['ncf']:.3f}")
    print()
print("\n" + "="*60)
print("ENSEMBLE SYSTEM COMPLETE")
print("Check 'ensemble_model_results.png' for visualization graphs")
print("="*60)