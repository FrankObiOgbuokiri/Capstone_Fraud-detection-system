import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads a CSV dataset from a given file path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.endswith(".csv"):
        raise ValueError("Only CSV files are supported.")
    
    df = pd.read_csv(file_path)
    print(f"CSV data loaded successfully with shape: {df.shape}")
    return df

def inspect_data(df):
    """Prints basic summary information about the dataset."""
    print("--- FIRST 5 ROWS ---")
    print(df.head())
    
    print("\n--- DATA INFO ---")
    print(df.info())
    
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe())

def feature_engineering(df):
    """
    Creates new features if needed.
    Such as Log-transforms for the 'Amount' column.
    """
    df_engineered = df.copy()
    
    # Log-transform Amount to handle skewed values
    if "Amount" in df_engineered.columns:
        import numpy as np
        df_engineered["Amount_log"] = np.log1p(df_engineered["Amount"])
        print("Created feature: 'Amount_log'")
        
    return df_engineered


def set_plot_style():
    """
    Sets the visual style for all EDA plots in this project.
    Call once at the top of the notebook, before any plotting functions.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)


def plot_class_distribution(df, class_col='Class'):
    """
    Prints class counts/percentages and plots the transaction class distribution
    on a linear scale.

    """
    fraud_counts = df[class_col].value_counts()
    total_records = len(df)
    pct_legit = (fraud_counts[0] / total_records) * 100
    pct_fraud = (fraud_counts[1] / total_records) * 100

    print(f"Legitimate Transactions: {fraud_counts[0]} ({pct_legit:.4f}%)")
    print(f"Fraudulent Transactions: {fraud_counts[1]} ({pct_fraud:.4f}%)")

    plt.figure(figsize=(6, 5))
    sns.countplot(x=class_col, data=df, hue=class_col, palette='Set1', legend=False)
    plt.title('Transaction Class Distribution\n(0: Legit, 1: Fraud)', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Number of Transactions')
    plt.xticks([0, 1], [f'Legitimate\n({pct_legit:.3f}%)', f'Fraudulent\n({pct_fraud:.3f}%)'])
    plt.tight_layout()
    plt.show()

    return {"fraud_counts": fraud_counts, "pct_legit": pct_legit, "pct_fraud": pct_fraud}


def plot_class_distribution_log(df, class_col='Class', pct_legit=None, pct_fraud=None):
    """
    Plots the transaction class distribution on a log scale, making the rare
    fraud class visible alongside the majority class.

    If pct_legit/pct_fraud aren't provided, they're computed from df.
    """
    if pct_legit is None or pct_fraud is None:
        fraud_counts = df[class_col].value_counts()
        total_records = len(df)
        pct_legit = (fraud_counts[0] / total_records) * 100
        pct_fraud = (fraud_counts[1] / total_records) * 100

    plt.figure(figsize=(8, 4))
    sns.countplot(y=class_col, data=df, hue=class_col, palette='Set1', legend=False)
    plt.xscale('log')
    plt.xlim(1, 10**6)
    plt.title('Transaction Class Distribution (Log Scale)\n(0: Legit, 1: Fraud)', fontsize=14)
    plt.ylabel('Class', fontsize=12)
    plt.xlabel('Number of Transactions (Log Scale)', fontsize=12)
    plt.yticks([0, 1], [f'Legitimate\n({pct_legit:.3f}%)', f'Fraudulent\n({pct_fraud:.3f}%)'])
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """
    Plots a full correlation heatmap across all features using a diverging
    coolwarm palette (red = positive, blue = negative correlation).

    Returns
    -------
    pd.DataFrame — the full correlation matrix
    """
    corr_matrix = df.corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False, fmt=".2f")
    plt.title('Correlation Heatmap of All Features', fontsize=16)
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plot_time_distribution(df, time_col='Time'):
    """
    Plots the distribution of transaction time (seconds elapsed).
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[time_col], bins=50, kde=True, color='teal')
    plt.title('Distribution of Transaction Time (Seconds)')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Density')
    plt.show()


def plot_amount_distribution(df, amount_col='Amount'):
    """
    Plots raw vs. log-transformed transaction amount distributions side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(df[amount_col], bins=50, kde=True, ax=axes[0], color='crimson')
    axes[0].set_title('Distribution of Amount')
    axes[0].set_xlabel('Amount')
    axes[0].set_ylabel('Density')

    sns.histplot(np.log1p(df[amount_col]), bins=50, kde=True, ax=axes[1], color='crimson')
    axes[1].set_title('Distribution of Log-Transformed Amount')
    axes[1].set_xlabel('Log(Amount + 1)')
    axes[1].set_ylabel('Density')

    plt.tight_layout()
    plt.show()


def plot_amount_boxplot_by_class(df, amount_col='Amount', class_col='Class', quantile_cutoff=0.95):
    """
    Boxplot of transaction amount by class, zoomed to below the given quantile
    to keep the box visible despite extreme outliers.
    """
    plt.figure(figsize=(8, 6))
    subset = df[df[amount_col] < df[amount_col].quantile(quantile_cutoff)]
    sns.boxplot(x=class_col, y=amount_col, data=subset, hue=class_col, palette='Set1', legend=False)
    plt.title(f'Boxplot of Transaction Amount by Class ({int(quantile_cutoff*100)}th Percentile)', fontsize=14)
    plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
    plt.xlabel('Transaction Class')
    plt.ylabel('Amount')
    plt.show()


def plot_feature_distributions_by_class(df, class_col='Class', selected_vars=None):
    """
    Plots KDE distributions for selected features, split by class, to visually
    compare how each feature's distribution differs between legitimate and
    fraudulent transactions.

    Defaults to V14, V17, V12, V10 — features that typically correlate
    strongly with the fraud class.
    """
    if selected_vars is None:
        selected_vars = ['V14', 'V17', 'V12', 'V10']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(selected_vars):
        sns.kdeplot(df[df[class_col] == 0][col], label='Legitimate', fill=True, color='blue', alpha=0.3, ax=axes[i])
        sns.kdeplot(df[df[class_col] == 1][col], label='Fraudulent', fill=True, color='red', alpha=0.3, ax=axes[i])
        axes[i].set_title(f'Distribution of {col} by Class')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def split_data(df, target_column='Class', test_size=0.2, random_state=42):
    """Splits the dataset into features (X) and target (y), then into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, columns_to_scale):
    """Scales specified numerical columns using StandardScaler."""
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale training data and apply the same scale to test data
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    print(f"Scaled columns: {columns_to_scale}")
    return X_train_scaled, X_test_scaled, scaler

