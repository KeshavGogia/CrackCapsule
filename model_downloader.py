import os
import requests
import hashlib
from pathlib import Path

def download_model_from_cloud():
    """
    Download the model file from cloud storage during deployment.
    This function will be called before loading the model.
    """
    model_path = 'Optimized_OmniCrack30k.pth'
    
    if os.path.exists(model_path):
        print("Model file already exists")
        return True
    
    print("Downloading model file...")
    model_url = os.environ.get('MODEL_DOWNLOAD_URL', '')
    
    if not model_url:
        print("MODEL_DOWNLOAD_URL environment variable not set")
        print("Please set the MODEL_DOWNLOAD_URL in Railway dashboard")
        return False
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r Downloading: {percent:.1f}%", end='', flush=True)
        
        print(f"\nModel downloaded successfully! Size: {downloaded / (1024*1024*1024):.2f} GB")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {str(e)}")
        return False

def verify_model_integrity():
    """Verify the model file integrity using checksum"""
    model_path = 'Optimized_OmniCrack30k.pth'
    
    if not os.path.exists(model_path):
        return False
    
    expected_hash = os.environ.get('MODEL_CHECKSUM', '')
    
    if expected_hash:
        print("🔍 Verifying model integrity...")
        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash == expected_hash:
            print("Model integrity verified")
            return True
        else:
            print("Model integrity check failed")
            return False
    
    return True  

if __name__ == "__main__":

    success = download_model_from_cloud()
    if success:
        verify_model_integrity()
