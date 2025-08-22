#!/usr/bin/env python3
"""
SAM2 Installation Test Script
Run this first to verify your SAM2 setup before using the UI
"""

import os
import sys

def test_sam2_installation():
    print("=" * 60)
    print("SAM2 Installation Test")
    print("=" * 60)
    
    # Test 1: Check if we're in the right directory
    print("\n1. Checking directory structure...")
    current_dir = os.getcwd()
    print(f"   Current directory: {current_dir}")
    
    expected_files = ['sam2', 'configs', 'checkpoints']
    missing_files = []
    
    for file in expected_files:
        if os.path.exists(file):
            print(f"   ✅ Found: {file}/")
        else:
            print(f"   ❌ Missing: {file}/")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n   ERROR: You need to run this from the segment-anything-2 directory!")
        print(f"   Make sure you have: {', '.join(missing_files)}")
        return False
    
    # Test 2: Check Python imports
    print("\n2. Testing Python imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   ❌ PyTorch not installed!")
        print("   Install with: pip install torch torchvision")
        return False
    
    try:
        import sam2
        print("   ✅ SAM2 module found")
    except ImportError:
        print("   ❌ SAM2 not installed!")
        print("   Install with: pip install -e .")
        return False
    
    try:
        from sam2.build_sam import build_sam2_video_predictor
        print("   ✅ SAM2 build functions available")
    except ImportError as e:
        print(f"   ❌ SAM2 build functions not available: {e}")
        return False
    
    # Test 3: Check model checkpoints
    print("\n3. Checking model checkpoints...")
    checkpoint_dir = "checkpoints"
    checkpoints = [
        "sam2_hiera_small.pt",
        "sam2_hiera_base_plus.pt", 
        "sam2_hiera_large.pt"
    ]
    
    found_checkpoints = []
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"   ✅ Found: {checkpoint} ({size_mb:.1f} MB)")
            found_checkpoints.append(checkpoint)
        else:
            print(f"   ❌ Missing: {checkpoint}")
    
    if not found_checkpoints:
        print("\n   ERROR: No model checkpoints found!")
        print("   Download at least one model:")
        print("   curl -o checkpoints/sam2_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt")
        return False
    
    # Test 4: Check config files
    print("\n4. Checking configuration files...")
    config_dir = "configs"
    configs = [
        "sam2_hiera_s.yaml",
        "sam2_hiera_b+.yaml",
        "sam2_hiera_l.yaml"
    ]
    
    found_configs = []
    for config in configs:
        config_path = os.path.join(config_dir, config)  # Use 'config' not 'sam2'
        if os.path.exists(config_path):
            print(f"   ✅ Found: {config}")
            found_configs.append(config)
        else:
            print(f"   ❌ Missing: {config}")
    
    if not found_configs:
        print("\n   ERROR: No configuration files found!")
        print("   The configs/ directory should be part of the SAM2 repository")
        return False
    
    # Test 5: Try loading a model
    print("\n5. Testing model loading...")
    try:
        # Use the first available checkpoint and corresponding config
        checkpoint_map = {
            "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
            "sam2_hiera_large.pt": "sam2_hiera_l.yaml"
        }
        
        test_checkpoint = None
        test_config = None
        
        for checkpoint in found_checkpoints:
            config_name = checkpoint_map[checkpoint]
            if config_name in [os.path.basename(c) for c in found_configs]:
                test_checkpoint = os.path.join(checkpoint_dir, checkpoint)
                test_config = os.path.join(config_dir, config_name)
                break
        
        if test_checkpoint and test_config:
            print(f"   Testing with: {os.path.basename(test_checkpoint)}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = build_sam2_video_predictor(test_config, test_checkpoint, device=device)
            
            print(f"   ✅ Model loaded successfully on {device.upper()}!")
            print(f"   ✅ Ready for video segmentation!")
            
        else:
            print("   ❌ Could not find matching checkpoint and config")
            return False
            
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("\n   This could be due to:")
        print("   - Insufficient GPU memory (try CPU: device='cpu')")
        print("   - Corrupted checkpoint file")
        print("   - Missing dependencies")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("Your SAM2 installation is working correctly!")
    print("You can now run the UI with: python sam2_ui.py")
    print("=" * 60)
    
    return True

def main():
    success = test_sam2_installation()
    if not success:
        print("\n" + "=" * 60)
        print("❌ SETUP INCOMPLETE")
        print("Please fix the issues above before running the UI")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()