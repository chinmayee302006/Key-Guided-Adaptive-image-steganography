# Key-Guided Adaptive Multi-Bit LSB Image Steganography Using Local Entropy and Channel Optimization

## Abstract
This paper presents an advanced image steganography system that enhances embedding capacity and security while maintaining high imperceptibility. By leveraging local Shannon entropy for adaptive bit allocation, the system dynamically assigns 1 to 3 bits per pixel based on the complexity of local image blocks. A dual randomization scheme, guided by a secret key, shuffles pixel indices and pseudo-randomly selects color channels. Furthermore, channel selection is optimized using local variance within the upper bits to ensure robustness. Experimental results demonstrate that the proposed method significantly increases embedding capacity (up to 2.44 bits per pixel) compared to standard 1-bit LSB (1.0 bits per pixel) while maintaining a Peak Signal-to-Noise Ratio (PSNR) well above the threshold of human perception.

## Introduction
Steganography is the art and science of hiding information in digital media such that the existence of the message is concealed. Digital images are popular cover media due to their high redundancy. The Least Significant Bit (LSB) method is a common technique, but basic implementations often suffer from low capacity and vulnerability to statistical analysis. Adaptive steganography aims to hide data in complex or "noisy" regions where modifications are less likely to be detected by the human visual system (HVS) or steganalysis tools. This research proposes a method that combines entropy-based adaptivity, key-guided randomization, and local variance-based channel optimization.

## Methodology
The proposed system follows an eight-step process:
1. **Preprocessing**: The RGB cover image is divided into non-overlapping 8x8 blocks.
2. **Local Entropy Calculation**: Shannon entropy is computed for each block using upper bits to ensure consistency between cover and stego images.
3. **Adaptive Bit Allocation**: Blocks are classified into low, medium, and high entropy regions, receiving 1, 2, or 3 bits respectively.
4. **Key-Based Randomization**: A secret key initializes a pseudo-random number generator to determine pixel and channel sequences.
5. **Channel Optimization**: Local variance is computed using robust upper bits. The system prefers channels with higher local variance to minimize visual distortion.
6. **Data Embedding**: Message bits are converted to binary and embedded into the determined LSB positions.
7. **Stego Image Generation**: The modified pixels are reconstructed to form the final stego image.
8. **Extraction Algorithm**: Using the same secret key, the entropy map and randomization sequences are regenerated to perfectly recover the hidden message.

## Results
The system was evaluated using a high-resolution cover image. Three methods were compared:
1. **Basic LSB**: Sequential 1-bit embedding in the Blue channel.
2. **Random LSB**: Key-randomized 1-bit embedding across all channels.
3. **Proposed Method**: Adaptive multi-bit embedding with channel optimization.

| Method | PSNR (dB) | MSE | Max Capacity (BPP) | Extraction |
| :--- | :--- | :--- | :--- | :--- |
| Basic LSB | 81.93 | 0.000417 | 1.0000 | SUCCESS |
| Random LSB | 81.78 | 0.000431 | 1.0000 | SUCCESS |
| Proposed | 74.83 | 0.002138 | 2.4439 | SUCCESS |

The proposed method achieved a **144% increase in maximum embedding capacity** over standard LSB while maintaining a PSNR of 74.83 dB, which is far above the standard 40 dB benchmark for high-quality steganography.

## Conclusion
The Key-Guided Adaptive Multi-Bit LSB system effectively addresses the trade-offs between capacity, imperceptibility, and security. By hiding more data in high-entropy regions and utilizing dual randomization, the system offers a robust solution for secure data transmission. Future work could explore the integration of error-correcting codes and resistance to more advanced frequency-domain steganalysis.
