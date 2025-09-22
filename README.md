# My-Experiments
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, probplot

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
