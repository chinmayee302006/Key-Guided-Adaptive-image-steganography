"""
Steganography Decoder Module
Provides a command-line interface to extract hidden messages from stego images.

Usage:
    python decoder.py --image <stego_image_path> --key <secret_key> --method <method_type>

Methods:
    1: Basic LSB
    2: Random LSB
    3: Proposed Adaptive (Default)
"""

import argparse
import sys
import os
import cv2
from steganography import (
    extract_basic_lsb,
    extract_random_lsb,
    extract_adaptive
)

def main():
    parser = argparse.ArgumentParser(description="Extract hidden messages from stego images.")
    parser.add_argument("--image", required=True, help="Path to the stego image.")
    parser.add_argument("--key", required=True, help="Secret key used during embedding.")
    parser.add_argument("--method", type=int, default=3, choices=[1, 2, 3],
                        help="Embedding method used (1: Basic, 2: Random, 3: Adaptive).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        sys.exit(1)
        
    # Load image
    stego_img = cv2.imread(args.image)
    if stego_img is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)
        
    print(f"[INFO] Decoding image: {args.image}")
    print(f"[INFO] Using method: {args.method}")
    
    # Attempt extraction
    try:
        if args.method == 1:
            message = extract_basic_lsb(stego_img)
        elif args.method == 2:
            # Try key as integer first, if fails use as string
            try:
                key = int(args.key)
            except ValueError:
                key = args.key
            message = extract_random_lsb(stego_img, key)
        else:
            try:
                key = int(args.key)
            except ValueError:
                key = args.key
            message = extract_adaptive(stego_img, key)
            
        print("\n" + "="*30)
        print("EXTRACTED MESSAGE:")
        print("="*30)
        print(message)
        print("="*30)
        
    except Exception as e:
        print(f"[ERROR] Extraction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
