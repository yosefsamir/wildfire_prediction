from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Get the absolute path to the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/data directory
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # Go up two levels to project root
print(f"Project root directory: {project_root}")

# Construct the path to the data file
file_path = os.path.join(project_root, 'data', 'processed', 'california_wildfires.csv')
abs_file_path = os.path.abspath(file_path)
print(f"Absolute file path: {abs_file_path}")
print(f"File exists: {os.path.exists(abs_file_path)}")

# Try to list files in the target directory
target_dir = os.path.dirname(abs_file_path)
if os.path.exists(target_dir):
    print(f"Files in {target_dir}:")
    for file in os.listdir(target_dir):
        print(f"  - {file}")
else:
    print(f"Directory {target_dir} does not exist")



# Read the CSV file and check for any issues
try:
    df = pd.read_csv(file_path)
        
    # Handle missing values (drop rows with any missing values)
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Successfully loaded {len(df)} rows")
    
    # Calculate skewness of FRP (Fire Radiative Power)
    frp_skewness = stats.skew(df['frp'])
    print(f"FRP Skewness: {frp_skewness:.4f}")
    
    # Create histogram visualization for FRP
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with KDE
    sns.histplot(df['frp'], kde=True, color='orangered')
    
    # Add vertical line for mean
    plt.axvline(df['frp'].mean(), color='red', linestyle='--', label=f'Mean: {df["frp"].mean():.2f}')
    
    # Add vertical line for median
    plt.axvline(df['frp'].median(), color='green', linestyle='-.', label=f'Median: {df["frp"].median():.2f}')
    
    # Add title and labels with skewness information
    plt.title(f'Distribution of Fire Radiative Power (FRP)\nSkewness: {frp_skewness:.4f}', fontsize=14)
    plt.xlabel('Fire Radiative Power (FRP)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the figure
    os.makedirs(os.path.join(project_root, 'reports', 'figures'), exist_ok=True)
    plt.savefig(os.path.join(project_root, 'reports', 'figures', 'frp_distribution.png'), 
               bbox_inches='tight', dpi=300)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Create a log-transformed histogram if data is highly skewed
    if abs(frp_skewness) > 1.0:
        plt.figure(figsize=(10, 6))
        
        # Add small constant to handle zeros if present
        log_frp = np.log1p(df['frp'])
        log_frp_skewness = stats.skew(log_frp)
        
        # Plot histogram with KDE for log-transformed data
        sns.histplot(log_frp, kde=True, color='blue')
        
        # Add title and labels with skewness information
        plt.title(f'Log-Transformed Distribution of FRP\nSkewness: {log_frp_skewness:.4f}', fontsize=14)
        plt.xlabel('Log(FRP + 1)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(project_root, 'reports', 'figures', 'log_frp_distribution.png'), 
                   bbox_inches='tight', dpi=300)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        print(f"Log-transformed FRP Skewness: {log_frp_skewness:.4f}")
        
except Exception as e:
    print(f"Error loading CSV file or processing FRP data: {e}")
    raise



