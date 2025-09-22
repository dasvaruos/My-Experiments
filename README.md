# My-Experiments
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, probplot, skew, kurtosis, shapiro
import torch
import torch.nn as nn
import torch.optim as optim

def plot_histogram_density_qqplot(path, min_n=3):
    df = pd.read_excel(path)
    for col in df.columns:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            n = len(data)
            if n < min_n:
                print(f"Skipping column {col}: too few valid entries ({n})")
                continue
            
            plt.figure(figsize=(12, 5))
            
            # Histogram with density curve
            plt.subplot(1, 2, 1)
            count, bins, ignored = plt.hist(data, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
            
            mu, std = norm.fit(data)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'r-', linewidth=2, label='Normal density')
            
            plt.title(f'Histogram & Density Curve of {col}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            
            # Q-Q plot
            plt.subplot(1, 2, 2)
            probplot(data, dist="norm", plot=plt)
            plt.title(f'Q-Q Plot of {col}')
            
            plt.suptitle(f'Distribution plots for column: {col}')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
        except Exception as e:
            print(f"Could not plot {col}: {e}")

if __name__ == "__main__":
    plot_histogram_density_qqplot(r"C:\Users\Dell\OneDrive\Desktop\Blogs\Currency Data.xlsx")

def standardize(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean()
    sd = x.std(ddof=1)
    return (x - mu) / sd if sd > 1e-6 else (x - mu)

def extract_features(x):
    z = standardize(x)
    quantiles = np.quantile(z, np.linspace(0, 1, 20))
    features = np.concatenate([
        quantiles,
        [z.mean(), z.std(ddof=1), skew(z), kurtosis(z)]
    ])
    return features.astype(np.float32)  # 24-dimensional fixed feature vector

class NormalityMLP(nn.Module):
    def __init__(self, input_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def make_batch(batch_size, n, rng):
    X = []
    y = []
    for _ in range(batch_size):
        if rng.random() < 0.5:
            x = gen_normal(n, rng)
            label = 0
        else:
            x = gen_alt(n, rng)
            label = 1
        feats = extract_features(x)
        X.append(feats)
        y.append(label)
    X = torch.tensor(np.stack(X))
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X, y

def gen_normal(n, rng):
    mu = rng.normal(0.0, 5.0)
    sigma = np.exp(rng.normal(0.0, 0.5))
    return rng.normal(mu, sigma, size=n).astype(np.float32)

def gen_alt(n, rng):
    # Same as before: various non-normal simulators
    # (implement or copy your previous gen_alt code here)
    # This is a placeholder
    return rng.standard_t(df=3, size=n).astype(np.float32)

def train_model(steps=2000, batch_size=256, lr=1e-3, seed=42):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = NormalityMLP()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    model.train()
    n = 521  # training sample length (use typical or max column length)
    for step in range(steps):
        Xb, yb = make_batch(batch_size, n, rng)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f'Step {step} train loss {loss.item():.4f}')
    return model

@torch.no_grad()
def score_sample(model, x):
    feats = torch.tensor(extract_features(x)[None, :])
    return float(model(feats).item())

def calibrate_null(model, n, M=20000, seed=123):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(M):
        x = gen_normal(n, rng)
        s = score_sample(model, x)
        scores.append(s)
    return np.array(scores)

def empirical_pvalue(score, null_scores):
    r = np.sum(null_scores >= score)
    M = len(null_scores)
    return (r + 1) / (M + 1)

def decision_threshold(null_scores, alpha=0.05):
    return np.quantile(null_scores, 1-alpha, method="higher")

def process_excel_and_test(path, model, null_scores, alpha=0.05, min_n=20):
    df = pd.read_excel(path)
    results = []
    for col in df.columns:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(data) < min_n:
                print(f"Skipping column {col}: insufficient size {len(data)}")
                continue
            s_obs = score_sample(model, data)
            p = empirical_pvalue(s_obs, null_scores)
            results.append({'column': col, 'score': s_obs, 'p_value': p, 'normal?': p > alpha})
        except Exception as e:
            print(f"Error processing {col}: {e}")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    n = 521  # or max column size expected
    model = train_model()
    null_scores = calibrate_null(model, n=n)
    process_excel_and_test(r"C:\Users\Dell\OneDrive\Desktop\Blogs\Currency Data.xlsx", model, null_scores)

def process_excel_shapiro_test(path, alpha=0.05, min_n=3):
    df = pd.read_excel(path)
    results = []
    for col in df.columns:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna().values
            n = len(data)
            if n < min_n:
                print(f"Skipping column {col}: too few valid entries ({n})")
                continue
            stat, p = shapiro(data)
            reject = p <= alpha
            results.append({'column': col, 'n': n, 'statistic': stat, 'p_value': p, 'normal?': not reject})
        except Exception as e:
            print(f"Could not process {col}: {e}")
    print("\nShapiro–Wilk Normality Test Results:")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    process_excel_shapiro_test(r"C:\Users\Dell\OneDrive\Desktop\Blogs\Currency Data.xlsx", alpha=0.05)
