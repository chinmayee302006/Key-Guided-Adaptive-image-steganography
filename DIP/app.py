import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from steganography import (
    embed_basic_lsb, extract_basic_lsb,
    embed_random_lsb, extract_random_lsb,
    embed_adaptive, extract_adaptive,
    compute_psnr, compute_mse, compute_max_capacity_bpp,
    compute_entropy_map, adaptive_bit_allocation
)

# Set page config
st.set_page_config(
    page_title="Key-Guided Adaptive Steganography",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4451;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def main():
    st.title(" Key-Guided Adaptive Multi-Bit Steganography")
    st.markdown("---")

    sidebar = st.sidebar
    sidebar.title("Navigation")
    app_mode = sidebar.radio("Select Action", ["Encode Message", "Decode Message", "About Research"])

    if app_mode == "Encode Message":
        st.header(" Encode a Secret Message")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. Configuration")
            cover_file = st.file_uploader("Upload Cover Image (PNG recommended)", type=["png", "jpg", "jpeg"])
            secret_message = st.text_area("Secret Message", placeholder="Enter the message you want to hide...")
            secret_key = st.text_input("Secret Key", type="password", placeholder="Enter a key for randomization")
            
            method = st.selectbox("Embedding Method", 
                                ["Proposed Adaptive (Entropy + Channel Opt.)", 
                                 "Random LSB (Key-Randomized)", 
                                 "Basic LSB (Sequential)"])
            
            if st.button("Run Steganography"):
                if not cover_file:
                    st.error("Please upload a cover image!")
                elif not secret_message:
                    st.error("Please enter a secret message!")
                elif not secret_key:
                    st.error("Please enter a secret key!")
                else:
                    # Processing
                    with st.spinner("Processing image..."):
                        # Load image
                        pil_img = Image.open(cover_file).convert("RGB")
                        cover_img = pil_to_cv2(pil_img)
                        
                        # Ensure dimensions are multiples of 8 for adaptive method
                        h, w = cover_img.shape[:2]
                        h = (h // 8) * 8
                        w = (w // 8) * 8
                        cover_img = cv2.resize(cover_img, (w, h))
                        
                        try:
                            if "Proposed" in method:
                                stego_img, used_bits, total_cap = embed_adaptive(cover_img, secret_message, secret_key)
                                max_bpp = compute_max_capacity_bpp(total_cap, cover_img.shape)
                            elif "Random" in method:
                                stego_img, used_bits, total_cap = embed_random_lsb(cover_img, secret_message, secret_key)
                                max_bpp = 1.0
                            else:
                                stego_img, used_bits, total_cap = embed_basic_lsb(cover_img, secret_message)
                                max_bpp = 1.0
                            
                            psnr = compute_psnr(cover_img, stego_img)
                            mse = compute_mse(cover_img, stego_img)
                            
                            st.session_state['stego_img'] = stego_img
                            st.session_state['psnr'] = psnr
                            st.session_state['mse'] = mse
                            st.session_state['max_bpp'] = max_bpp
                            st.session_state['success'] = True
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        with col2:
            if 'success' in st.session_state:
                st.subheader("2. Results & Metrics")
                
                # Show images
                st.image(cv2_to_pil(st.session_state['stego_img']), caption="Generated Stego Image", use_container_width=True)
                
                # Metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("PSNR (dB)", f"{st.session_state['psnr']:.2f}")
                m_col2.metric("MSE", f"{st.session_state['mse']:.5f}")
                m_col3.metric("Max Capacity", f"{st.session_state['max_bpp']:.2f} BPP")
                
                # Download
                result_pil = cv2_to_pil(st.session_state['stego_img'])
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label=" Download Stego Image",
                    data=byte_im,
                    file_name="stego_image.png",
                    mime="image/png"
                )
            else:
                st.info("Upload an image and click 'Run Steganography' to see results.")

    elif app_mode == "Decode Message":
        st.header("Decode a Secret Message")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            stego_file = st.file_uploader("Upload Stego Image", type=["png"])
            decode_key = st.text_input("Secret Key", type="password")
            decode_method = st.selectbox("Expected Method", 
                                       ["Proposed Adaptive", "Random LSB", "Basic LSB"])
            
            if st.button("Extract Message"):
                if not stego_file:
                    st.error("Please upload the stego image!")
                elif not decode_key and "Basic" not in decode_method:
                    st.error("Please enter the secret key!")
                else:
                    with st.spinner("Extracting..."):
                        pil_img = Image.open(stego_file).convert("RGB")
                        stego_img = pil_to_cv2(pil_img)
                        
                        try:
                            if "Proposed" in decode_method:
                                message = extract_adaptive(stego_img, decode_key)
                            elif "Random" in decode_method:
                                message = extract_random_lsb(stego_img, decode_key)
                            else:
                                message = extract_basic_lsb(stego_img)
                            
                            st.session_state['decoded_msg'] = message
                        except Exception as e:
                            st.error(f"Extraction failed. Incorrect key or method? Error: {str(e)}")

        with col2:
            if 'decoded_msg' in st.session_state:
                st.subheader("Extracted Message")
                st.success("Message recovered successfully!")
                st.text_area("Message Content", value=st.session_state['decoded_msg'], height=300)
            else:
                st.info("Upload the stego image and provide the key to extract the hidden message.")

    elif app_mode == "About Research":
        st.header("Research Methodology")
        st.markdown("""
        ### Key-Guided Adaptive Multi-Bit LSB Steganography
        
        This project implements a state-of-the-art steganography system designed for high capacity and security.
        
        #### Core Components:
        1. **Adaptive Bit Allocation**: Calculates Shannon entropy for 8x8 blocks. Texture-heavy regions receive more bits (up to 3) while smooth regions receive less (1 bit).
        2. **Key randomization**: Uses SHA-256 derived seeds to randomize pixel traversal and channel selection (R/G/B).
        3. **Channel Optimization**: Analyzes local variance on upper bits to hide data where it is most visually resistant to detection.
        
        #### Performance Benchmarks:
        - **Capacity**: Up to 2.44 bits per pixel (vs 1.0 in standard LSB).
        - **Quality**: PSNR remains high (>70dB for small messages).
        - **Security**: Robust against visual attacks and basic histogram analysis due to dual randomization.
        """)
        
        if os.path.exists("results/metrics_comparison.png"):
            st.image("results/metrics_comparison.png", caption="Comparative Performance Metrics")

if __name__ == "__main__":
    main()
