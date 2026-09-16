 
import pandas as pd
import urllib.request
import os

def download_nsl_kdd():
    """Download NSL-KDD dataset"""
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # URLs for NSL-KDD dataset
    urls = {
        'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv',
        'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv'
    }
    
    # Column names for NSL-KDD
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # Download training data
    print("Downloading NSL-KDD Training dataset...")
    train_df = pd.read_csv(urls['train'], header=None, names=columns)
    train_df.to_csv('data/raw/KDDTrain.csv', index=False)
    print(f"Training data saved: {train_df.shape}")
    
    # Download test data
    print("Downloading NSL-KDD Test dataset...")
    test_df = pd.read_csv(urls['test'], header=None, names=columns)
    test_df.to_csv('data/raw/KDDTest.csv', index=False)
    print(f"Test data saved: {test_df.shape}")
    
    return train_df, test_df

if __name__ == "__main__":
    train, test = download_nsl_kdd()
    print("\n✅ Dataset downloaded successfully!")
    print(f"Training samples: {len(train)}")
    print(f"Test samples: {len(test)}")