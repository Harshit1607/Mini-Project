"""
Download PhysioNet Sleep-Accel Dataset
This script downloads the required dataset for the sleep classification project
"""

import os
import urllib.request
import zipfile
import shutil

def download_dataset():
    """Download and extract PhysioNet sleep-accel dataset"""
    
    print("="*60)
    print("DOWNLOADING PHYSIONET SLEEP-ACCEL DATASET")
    print("="*60)
    
    # Create data directory
    data_dir = './sleep-accel-data'
    if os.path.exists(data_dir):
        print(f"\nData directory '{data_dir}' already exists.")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Using existing data.")
            return
        shutil.rmtree(data_dir)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Dataset URL
    dataset_url = "https://physionet.org/static/published-projects/sleep-accel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip"
    zip_path = "sleep-accel.zip"
    
    print(f"\nDownloading dataset from PhysioNet...")
    print("This may take several minutes (550 MB)...")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end='')
        
        urllib.request.urlretrieve(dataset_url, zip_path, reporthook=report_progress)
        print("\n\nDownload completed!")
        
        # Extract files
        print("\nExtracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temporary directory
            temp_dir = './temp_extract'
            zip_ref.extractall(temp_dir)
            
            # Move files from nested directory to our data directory
            extracted_base = os.path.join(temp_dir, 'motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0')
            
            # Copy all .txt files to data directory
            for file in os.listdir(extracted_base):
                if file.endswith('.txt'):
                    src = os.path.join(extracted_base, file)
                    dst = os.path.join(data_dir, file)
                    shutil.copy2(src, dst)
            
            # Clean up
            shutil.rmtree(temp_dir)
        
        # Remove zip file
        os.remove(zip_path)
        
        print("Extraction completed!")
        
        # Count files
        accel_files = len([f for f in os.listdir(data_dir) if f.endswith('_acceleration.txt')])
        print(f"\n✓ Successfully downloaded data for {accel_files} subjects")
        print(f"✓ Data saved to: {os.path.abspath(data_dir)}")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nMANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Visit: https://physionet.org/content/sleep-accel/1.0.0/")
        print("2. Click 'Download the ZIP file'")
        print("3. Extract the contents")
        print(f"4. Copy all .txt files to: {os.path.abspath(data_dir)}")
        return False
    
    print("\n" + "="*60)
    print("DATASET READY FOR TRAINING!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = download_dataset()
    
    if success:
        print("\nNext step: Run the training script")
        print("  python sleep_paralysis_rf_classifier.py")
    else:
        print("\nPlease download the dataset manually and try again.")