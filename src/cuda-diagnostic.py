import torch
import sys
import subprocess
import os


def check_cuda_availability():
    print("=== Sistema Python ===")
    print(f"Python version: {sys.version}")
    
    print("\n=== Versiones de PyTorch ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    
    print("\n=== NVIDIA-SMI ===")
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        print(nvidia_smi.decode())
    except subprocess.CalledProcessError:
        print("nvidia-smi no está disponible o no se encontraron GPUs")
    
    print("\n=== CUDA_HOME ===")
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home}")


if __name__ == "__main__":
    check_cuda_availability()
