"""
Key-Guided Adaptive Multi-Bit LSB Image Steganography
Using Local Entropy and Channel Optimization

This module implements three steganography methods:
1. Basic LSB (sequential, 1-bit)
2. Random LSB (key-randomized, 1-bit)
3. Proposed Adaptive Method (entropy-based multi-bit with channel optimization)


"""

import numpy as np
import cv2
from collections import Counter
import hashlib


# ============================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================

def text_to_binary(text):
    """Convert a text string to its binary representation.
    
    Parameters:
        text (str): Input text string
        
    Returns:
        str: Binary string representation (8 bits per character)
    """
    return ''.join(format(ord(c), '08b') for c in text)


def binary_to_text(binary_str):
    """Convert a binary string back to text.
    
    Parameters:
        binary_str (str): Binary string (must be multiple of 8 bits)
        
    Returns:
        str: Decoded text string
    """
    chars = []
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return ''.join(chars)


def int_to_bin(n, bits=32):
    """Convert integer to fixed-width binary string."""
    return format(n, f'0{bits}b')


def bin_to_int(b):
    """Convert binary string to integer."""
    return int(b, 2)


def clamp(value, low=0, high=255):
    """Clamp a pixel value to valid range [0, 255]."""
    return max(low, min(high, int(value)))


# ============================================================
# SECTION 2: ENTROPY CALCULATION
# ============================================================

def compute_block_entropy(block):
    """
    Compute Shannon entropy of an image block.
    
    Formula: H = -Σ p(i) * log₂(p(i))
    
    Parameters:
        block (np.ndarray): 2D array representing a grayscale block
        
    Returns:
        float: Shannon entropy value (higher = more complex)
    """
    flat = block.flatten()
    total = len(flat)
    if total == 0:
        return 0.0
    counts = Counter(flat)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def compute_entropy_map(image, block_size=8):
    """
    Compute entropy map for the entire image using non-overlapping blocks.
    
    Uses upper bits (bits 3-7) for robustness — ensures identical entropy
    maps for both cover and stego images during extraction.
    
    Parameters:
        image (np.ndarray): BGR image (H x W x 3)
        block_size (int): Size of non-overlapping blocks (default: 8)
        
    Returns:
        np.ndarray: 2D entropy map (blocks_h x blocks_w)
    """
    # Shift bits BEFORE grayscale conversion to ensure robustness
    # This ensures LSB changes in BGR don't affect the Gray upper bits
    img_upper = (image >> 3) << 3
    gray_upper = cv2.cvtColor(img_upper, cv2.COLOR_BGR2GRAY)
    
    h, w = gray_upper.shape
    blocks_h = h // block_size
    blocks_w = w // block_size
    
    entropy_map = np.zeros((blocks_h, blocks_w))
    for i in range(blocks_h):
        for j in range(blocks_w):
            block = gray_upper[i*block_size:(i+1)*block_size,
                               j*block_size:(j+1)*block_size]
            entropy_map[i, j] = compute_block_entropy(block)
    
    return entropy_map


# ============================================================
# SECTION 3: ADAPTIVE BIT ALLOCATION
# ============================================================

def adaptive_bit_allocation(entropy_map, low_thresh=1.5, high_thresh=3.5):
    """
    Assign embedding bit depth based on block entropy.
    
    Classification:
        - Low entropy  (H < low_thresh)  → 1 bit (smooth regions)
        - Medium entropy                 → 2 bits
        - High entropy (H >= high_thresh) → 3 bits (complex regions)
    
    Parameters:
        entropy_map (np.ndarray): 2D array of block entropy values
        low_thresh (float): Threshold for low/medium boundary
        high_thresh (float): Threshold for medium/high boundary
        
    Returns:
        np.ndarray: 2D bit allocation map (values: 1, 2, or 3)
    """
    bit_map = np.ones_like(entropy_map, dtype=int)
    bit_map[entropy_map >= low_thresh] = 2
    bit_map[entropy_map >= high_thresh] = 3
    return bit_map


def get_pixel_bits(bit_map, block_size, row, col):
    """
    Get the bit allocation for a specific pixel based on its block.
    
    Parameters:
        bit_map (np.ndarray): Block-level bit allocation map
        block_size (int): Block size used for entropy computation
        row (int): Pixel row index
        col (int): Pixel column index
        
    Returns:
        int: Number of bits to embed (1, 2, or 3)
    """
    br = row // block_size
    bc = col // block_size
    if br < bit_map.shape[0] and bc < bit_map.shape[1]:
        return bit_map[br, bc]
    return 1  # Default to 1 bit for boundary pixels


# ============================================================
# SECTION 4: KEY-BASED RANDOMIZATION
# ============================================================

def key_to_seed(key):
    """
    Convert a key (string or int) to a deterministic integer seed.
    
    Uses SHA-256 hashing for uniform distribution.
    
    Parameters:
        key: Secret key (str or int)
        
    Returns:
        int: Deterministic seed value
    """
    key_str = str(key)
    hash_val = hashlib.sha256(key_str.encode()).hexdigest()
    return int(hash_val[:8], 16)


def generate_pixel_order(key, total_pixels):
    """
    Generate a randomized pixel index order using the secret key.
    
    Parameters:
        key: Secret key for reproducibility
        total_pixels (int): Total number of pixels in the image
        
    Returns:
        np.ndarray: Shuffled array of pixel indices
    """
    seed = key_to_seed(key)
    rng = np.random.RandomState(seed)
    indices = np.arange(total_pixels)
    rng.shuffle(indices)
    return indices


def generate_channel_sequence(key, length):
    """
    Generate a pseudo-random channel selection sequence.
    
    Parameters:
        key: Secret key
        length (int): Number of channel selections needed
        
    Returns:
        np.ndarray: Array of channel indices (0=B, 1=G, 2=R in OpenCV)
    """
    seed = key_to_seed(str(key) + "_channel")
    rng = np.random.RandomState(seed)
    return rng.randint(0, 3, size=length)


# ============================================================
# SECTION 5: CHANNEL OPTIMIZATION
# ============================================================

def compute_local_variance(image, row, col, window=3):
    """
    Compute local variance for each channel around a pixel.
    Uses upper bits only (bits 3-7) to ensure consistency between
    cover and stego images.
    
    Parameters:
        image (np.ndarray): BGR image
        row, col (int): Pixel coordinates
        window (int): Window size for variance computation
        
    Returns:
        np.ndarray: Variance for each channel [B, G, R]
    """
    h, w = image.shape[:2]
    half = window // 2
    r_start = max(0, row - half)
    r_end = min(h, row + half + 1)
    c_start = max(0, col - half)
    c_end = min(w, col + half + 1)
    
    # Use upper bits only for robustness (identical for cover and stego)
    patch = image[r_start:r_end, c_start:c_end].astype(np.float64)
    patch_upper = np.floor(patch / 8.0) * 8.0
    variances = np.var(patch_upper, axis=(0, 1))
    return variances


def select_optimal_channel(image, row, col, key_channel):
    """
    Select the optimal channel for embedding.
    
    Uses a hybrid approach: key-based selection weighted by local variance.
    Channels with higher variance are preferred (distortion is less noticeable).
    
    Parameters:
        image (np.ndarray): BGR image
        row, col (int): Pixel coordinates
        key_channel (int): Key-suggested channel
        
    Returns:
        int: Selected channel index (0=B, 1=G, 2=R)
    """
    variances = compute_local_variance(image, row, col)
    # Prefer the channel with maximum variance (most complex)
    max_var_channel = np.argmax(variances)
    
    # If the key-suggested channel has reasonable variance, use it
    # Otherwise, use the max variance channel for better imperceptibility
    if variances[key_channel] >= 0.5 * variances[max_var_channel]:
        return key_channel
    return int(max_var_channel)


# ============================================================
# SECTION 6: EMBEDDING METHODS
# ============================================================

def _prepare_message_bits(message):
    """Convert message to binary with 32-bit length header."""
    msg_binary = text_to_binary(message)
    msg_len = len(msg_binary)
    header = int_to_bin(msg_len, 32)
    return header + msg_binary


def embed_basic_lsb(cover_image, message):
    """
    Basic LSB Steganography (Method 1 — Baseline).
    
    Embeds data sequentially into the LSB of the Blue channel.
    No randomization, no adaptive allocation.
    
    Parameters:
        cover_image (np.ndarray): BGR cover image
        message (str): Secret text message
        
    Returns:
        tuple: (stego_image, capacity_used, total_capacity)
    """
    stego = cover_image.copy()
    h, w = stego.shape[:2]
    total_capacity = h * w  # 1 bit per pixel, blue channel
    
    all_bits = _prepare_message_bits(message)
    if len(all_bits) > total_capacity:
        raise ValueError(f"Message too long! Need {len(all_bits)} bits, capacity: {total_capacity}")
    
    bit_idx = 0
    for row in range(h):
        for col in range(w):
            if bit_idx >= len(all_bits):
                break
            # Embed in Blue channel (index 0 in BGR)
            pixel_val = int(stego[row, col, 0])
            pixel_val = (pixel_val & 0xFE) | int(all_bits[bit_idx])
            stego[row, col, 0] = clamp(pixel_val)
            bit_idx += 1
        if bit_idx >= len(all_bits):
            break
    
    return stego, len(all_bits), total_capacity


def extract_basic_lsb(stego_image):
    """
    Extract message from Basic LSB stego image.
    
    Parameters:
        stego_image (np.ndarray): BGR stego image
        
    Returns:
        str: Extracted text message
    """
    h, w = stego_image.shape[:2]
    
    # Extract 32-bit header first
    bits = []
    bit_idx = 0
    for row in range(h):
        for col in range(w):
            if bit_idx >= 32:
                break
            bits.append(str(stego_image[row, col, 0] & 1))
            bit_idx += 1
        if bit_idx >= 32:
            break
    
    msg_len = bin_to_int(''.join(bits[:32]))
    total_bits = 32 + msg_len
    
    # Extract remaining message bits
    bits = []
    bit_idx = 0
    for row in range(h):
        for col in range(w):
            if bit_idx >= total_bits:
                break
            bits.append(str(stego_image[row, col, 0] & 1))
            bit_idx += 1
        if bit_idx >= total_bits:
            break
    
    msg_bits = ''.join(bits[32:32+msg_len])
    return binary_to_text(msg_bits)


def embed_random_lsb(cover_image, message, key):
    """
    Random LSB Steganography (Method 2).
    
    Embeds data using key-based random pixel order and random channel selection.
    Still uses 1-bit LSB, but with dual randomization for security.
    
    Parameters:
        cover_image (np.ndarray): BGR cover image
        message (str): Secret text message
        key: Secret key for randomization
        
    Returns:
        tuple: (stego_image, capacity_used, total_capacity)
    """
    stego = cover_image.copy()
    h, w = stego.shape[:2]
    total_pixels = h * w
    total_capacity = total_pixels  # 1 bit per pixel
    
    all_bits = _prepare_message_bits(message)
    if len(all_bits) > total_capacity:
        raise ValueError(f"Message too long! Need {len(all_bits)} bits, capacity: {total_capacity}")
    
    # Generate random pixel order and channel sequence
    pixel_order = generate_pixel_order(key, total_pixels)
    channel_seq = generate_channel_sequence(key, total_pixels)
    
    for i in range(len(all_bits)):
        px_idx = pixel_order[i]
        row = px_idx // w
        col = px_idx % w
        ch = channel_seq[i]
        
        pixel_val = int(stego[row, col, ch])
        pixel_val = (pixel_val & 0xFE) | int(all_bits[i])
        stego[row, col, ch] = clamp(pixel_val)
    
    return stego, len(all_bits), total_capacity


def extract_random_lsb(stego_image, key):
    """
    Extract message from Random LSB stego image.
    
    Parameters:
        stego_image (np.ndarray): BGR stego image
        key: Same secret key used during embedding
        
    Returns:
        str: Extracted text message
    """
    h, w = stego_image.shape[:2]
    total_pixels = h * w
    
    pixel_order = generate_pixel_order(key, total_pixels)
    channel_seq = generate_channel_sequence(key, total_pixels)
    
    # Extract 32-bit header
    header_bits = []
    for i in range(32):
        px_idx = pixel_order[i]
        row = px_idx // w
        col = px_idx % w
        ch = channel_seq[i]
        header_bits.append(str(stego_image[row, col, ch] & 1))
    
    msg_len = bin_to_int(''.join(header_bits))
    total_bits = 32 + msg_len
    
    # Extract all bits
    all_bits = []
    for i in range(total_bits):
        px_idx = pixel_order[i]
        row = px_idx // w
        col = px_idx % w
        ch = channel_seq[i]
        all_bits.append(str(stego_image[row, col, ch] & 1))
    
    msg_bits = ''.join(all_bits[32:32+msg_len])
    return binary_to_text(msg_bits)


def embed_adaptive(cover_image, message, key, block_size=8,
                   low_thresh=1.5, high_thresh=3.5):
    """
    Proposed Method: Key-Guided Adaptive Multi-Bit LSB Steganography
    with Local Entropy and Channel Optimization (Method 3).
    
    Features:
        - Entropy-based adaptive bit allocation (1-3 bits per pixel)
        - Key-based random pixel ordering
        - Channel optimization using local variance
        - Dual randomization for enhanced security
    
    Parameters:
        cover_image (np.ndarray): BGR cover image
        message (str): Secret text message
        key: Secret key for reproducibility
        block_size (int): Block size for entropy computation
        low_thresh (float): Low entropy threshold
        high_thresh (float): High entropy threshold
        
    Returns:
        tuple: (stego_image, capacity_used, total_capacity)
    """
    stego = cover_image.copy()
    h, w = stego.shape[:2]
    total_pixels = h * w
    
    # Step 1: Compute entropy map and bit allocation
    entropy_map = compute_entropy_map(cover_image, block_size)
    bit_map = adaptive_bit_allocation(entropy_map, low_thresh, high_thresh)
    
    # Step 2: Calculate total capacity
    total_capacity = 0
    for px_idx in range(total_pixels):
        row = px_idx // w
        col = px_idx % w
        total_capacity += get_pixel_bits(bit_map, block_size, row, col)
    
    # Step 3: Prepare message
    all_bits = _prepare_message_bits(message)
    if len(all_bits) > total_capacity:
        raise ValueError(f"Message too long! Need {len(all_bits)} bits, capacity: {total_capacity}")
    
    # Step 4: Generate random pixel order and channel sequence
    pixel_order = generate_pixel_order(key, total_pixels)
    channel_seq = generate_channel_sequence(key, total_pixels)
    
    # Step 5: Embed bits adaptively
    bit_idx = 0
    for i in range(total_pixels):
        if bit_idx >= len(all_bits):
            break
        
        px_idx = pixel_order[i]
        row = px_idx // w
        col = px_idx % w
        
        # Get adaptive bit count for this pixel's block
        n_bits = get_pixel_bits(bit_map, block_size, row, col)
        
        # Select optimal channel (uses upper bits — deterministic)
        ch = select_optimal_channel(cover_image, row, col, channel_seq[i])
        
        # Get bits to embed (pad with zeros if fewer than n_bits remain)
        bits_to_embed = all_bits[bit_idx:bit_idx + n_bits]
        actual_bits = len(bits_to_embed)
        
        if actual_bits == 0:
            break
        
        # Pad to full n_bits so we always embed exactly n_bits
        padded_bits = bits_to_embed.ljust(n_bits, '0')
        
        # Embed n_bits into the LSBs of the selected channel
        pixel_val = int(stego[row, col, ch])
        
        # Create mask to clear the lower n_bits
        mask = (0xFF << n_bits) & 0xFF
        # Convert bits string to integer value
        embed_val = int(padded_bits, 2)
        
        # Apply embedding
        new_val = (pixel_val & mask) | embed_val
        stego[row, col, ch] = clamp(new_val)
        
        bit_idx += actual_bits
    
    return stego, len(all_bits), total_capacity


def extract_adaptive(stego_image, key, block_size=8,
                     low_thresh=1.5, high_thresh=3.5):
    """
    Extract message from Proposed Adaptive Method stego image.
    
    Regenerates the same entropy map, pixel order, and channel sequence
    using the provided key to accurately extract embedded data.
    
    Parameters:
        stego_image (np.ndarray): BGR stego image
        key: Same secret key used during embedding
        block_size (int): Same block size used during embedding
        low_thresh (float): Same low entropy threshold
        high_thresh (float): Same high entropy threshold
        
    Returns:
        str: Extracted text message
    """
    h, w = stego_image.shape[:2]
    total_pixels = h * w
    
    # Recompute entropy map (uses upper bits — identical to cover image)
    entropy_map = compute_entropy_map(stego_image, block_size)
    bit_map = adaptive_bit_allocation(entropy_map, low_thresh, high_thresh)
    
    # Regenerate same random sequences
    pixel_order = generate_pixel_order(key, total_pixels)
    channel_seq = generate_channel_sequence(key, total_pixels)
    
    # Phase 1: Extract header (32 bits) to determine message length
    header_bits = []
    header_idx = 0
    px_count = 0
    
    while header_idx < 32 and px_count < total_pixels:
        px_idx = pixel_order[px_count]
        row = px_idx // w
        col = px_idx % w
        n_bits = get_pixel_bits(bit_map, block_size, row, col)
        ch = select_optimal_channel(stego_image, row, col, channel_seq[px_count])
        
        pixel_val = int(stego_image[row, col, ch])
        
        # Extract n_bits from LSBs
        for b in range(n_bits - 1, -1, -1):
            if header_idx < 32:
                header_bits.append(str((pixel_val >> b) & 1))
                header_idx += 1
        
        px_count += 1
    
    msg_len = bin_to_int(''.join(header_bits[:32]))
    total_bits = 32 + msg_len
    
    # Phase 2: Extract all bits (restart from beginning)
    all_bits = []
    bit_count = 0
    
    for i in range(total_pixels):
        if bit_count >= total_bits:
            break
        
        px_idx = pixel_order[i]
        row = px_idx // w
        col = px_idx % w
        n_bits = get_pixel_bits(bit_map, block_size, row, col)
        ch = select_optimal_channel(stego_image, row, col, channel_seq[i])
        
        pixel_val = int(stego_image[row, col, ch])
        
        for b in range(n_bits - 1, -1, -1):
            if bit_count < total_bits:
                all_bits.append(str((pixel_val >> b) & 1))
                bit_count += 1
    
    msg_bits = ''.join(all_bits[32:32+msg_len])
    return binary_to_text(msg_bits)


# ============================================================
# SECTION 7: PERFORMANCE METRICS
# ============================================================

def compute_mse(original, stego):
    """
    Compute Mean Squared Error between original and stego images.
    
    MSE = (1/MN) * Σ Σ (I(i,j) - K(i,j))²
    
    Parameters:
        original (np.ndarray): Original cover image
        stego (np.ndarray): Stego image
        
    Returns:
        float: MSE value (lower is better)
    """
    diff = original.astype(np.float64) - stego.astype(np.float64)
    return np.mean(diff ** 2)


def compute_psnr(original, stego):
    """
    Compute Peak Signal-to-Noise Ratio.
    
    PSNR = 10 * log10(255² / MSE)
    
    Parameters:
        original (np.ndarray): Original cover image
        stego (np.ndarray): Stego image
        
    Returns:
        float: PSNR in dB (higher is better, >40 dB considered good)
    """
    mse = compute_mse(original, stego)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


def compute_capacity_bpp(bits_used, image_shape):
    """
    Compute embedding capacity in bits per pixel.
    
    Parameters:
        bits_used (int): Number of bits embedded
        image_shape (tuple): Image shape (H, W, C)
        
    Returns:
        float: Bits per pixel
    """
    total_pixels = image_shape[0] * image_shape[1]
    return bits_used / total_pixels


def compute_max_capacity_bpp(total_capacity, image_shape):
    """Compute maximum capacity in bits per pixel."""
    total_pixels = image_shape[0] * image_shape[1]
    return total_capacity / total_pixels
