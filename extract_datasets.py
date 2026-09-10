import os
import sys
import shutil
import zipfile
import subprocess

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        # Try tar first as it's faster
        subprocess.run(['tar', '-xf', zip_path, '-C', extract_to], check=True)
    except Exception as e:
        print(f"tar failed: {e}. Falling back to zipfile...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    print(f"Done extracting {os.path.basename(zip_path)}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    downloads_dir = r"C:\Users\rishi\Downloads"
    
    # 1. Create structure
    dirs = [
        'lfw', 'nuaa', 'replayattack', 'celeba_spoof', 
        'rppg_kaggle', 'rppg_nfi', 'cmu_keystroke'
    ]
    for d in dirs:
        path = os.path.join(data_dir, d)
        os.makedirs(path, exist_ok=True)
        
    # 2. Add to gitignore
    gitignore_path = os.path.join(base_dir, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            content = f.read()
        if 'data/' not in content:
            with open(gitignore_path, 'a') as f:
                f.write('\ndata/\n')
    else:
        with open(gitignore_path, 'w') as f:
            f.write('data/\n')
            
    # 3. Extract datasets
    extract_zip(os.path.join(downloads_dir, 'lfw.zip'), os.path.join(data_dir, 'lfw'))
    extract_zip(os.path.join(downloads_dir, 'nua.zip'), os.path.join(data_dir, 'nuaa'))
    extract_zip(os.path.join(downloads_dir, 'replayattack.zip'), os.path.join(data_dir, 'replayattack'))
    extract_zip(os.path.join(downloads_dir, 'celeba.zip'), os.path.join(data_dir, 'celeba_spoof'))
    extract_zip(os.path.join(downloads_dir, 'rppg.zip'), os.path.join(data_dir, 'rppg_kaggle'))
    
    # 4. Copy CSV
    csv_src = os.path.join(downloads_dir, 'DSL-StrongPasswordData.csv')
    csv_dest = os.path.join(data_dir, 'cmu_keystroke', 'DSL-StrongPasswordData.csv')
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, csv_dest)
        print("Copied DSL-StrongPasswordData.csv")
    else:
        print(f"File not found: {csv_src}")
        
    # 5. Link NFI rPPG dataset (C:\Users\rishi\data\rppg -> data/rppg_nfi)
    nfi_src = r"C:\Users\rishi\data\rppg"
    nfi_dest = os.path.join(data_dir, 'rppg_nfi')
    
    # Remove empty dir created above
    if os.path.exists(nfi_dest) and os.path.isdir(nfi_dest) and not os.listdir(nfi_dest):
        os.rmdir(nfi_dest)
        
    if not os.path.exists(nfi_dest):
        if os.path.exists(nfi_src):
            try:
                # Use junction point on Windows
                subprocess.run(['cmd', '/c', 'mklink', '/J', nfi_dest, nfi_src], check=True)
                print(f"Created junction point for NFI dataset.")
            except Exception as e:
                print(f"Failed to create junction: {e}")
        else:
            print(f"Source NFI dataset not found: {nfi_src}")
            
if __name__ == '__main__':
    main()
