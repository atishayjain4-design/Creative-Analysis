import streamlit as st
import pandas as pd
import cv2
import numpy as np
import re
import os
import easyocr
from io import BytesIO
from collections import Counter

# --- 1. FEATURE EXTRACTION FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads OpenCV and EasyOCR models into memory."""
    print("Loading models...")
    # This file MUST be in your GitHub repository
    cascade_path = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path):
        st.error(f"Fatal Error: `{cascade_path}` not found.")
        st.error("Please download this file and upload it to your GitHub repository:")
        st.code("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        return None, None
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    print("Loading EasyOCR model (this may take a moment)...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    print("Models loaded successfully.")
    return face_cascade, ocr_reader

def analyze_image_features(image_bytes, face_cascade, ocr_reader):
    """
    Analyzes a single image (as bytes) and returns a dictionary of its features.
    """
    try:
        # Convert bytes to OpenCV image
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # --- CV Features ---
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        has_face = len(faces) > 0
        
        brightness = np.mean(gray)
        if brightness < 90: brightness_level = "Low (Dark)"
        elif brightness < 180: brightness_level = "Medium (Balanced)"
        else: brightness_level = "High (Bright)"

        # --- OCR Features ---
        results = ocr_reader.readtext(image_bytes, detail=0, paragraph=True)
        raw_text = " ".join(results)
        
        # Basic Cleaning
        cleaned_text = raw_text.upper()
        has_text = len(cleaned_text.strip()) > 5
        
        # --- SMART PRICE EXTRACTION LOGIC ---
        
        # 1. Hook Price Regex: Looks for "FROM ?29,999", "JUST ₹499", "STARTS @ 99"
        # Note: We explicitly look for '?' because OCR often mistakes ₹ for ?
        hook_price_regex = re.compile(r"((?:FROM|STARTS?|JUST|ONLY|NOW|AT|@)\s*(?:₹|\$|€|£|RS\.?|INR|\?)\s*[\d,.]+)")

        # 2. Loose Price Regex: Looks for "₹29,999" or "?29,999" without the hook words
        loose_price_regex = re.compile(r"((?:₹|\$|€|£|RS\.?|INR|\?)\s*[\d,.]+)")
        
        # 3. Offer Regex: Looks for "50% OFF", "SALE", "FREE"
        offer_regex = re.compile(r"(\d{1,2}\s?% (?:OFF)?|SALE|FREE SHIPPING|FREE|BOGO|DEAL|OFFER|FLAT \d+%)")
        
        callout_type = "None"
        extracted_price = None
        extracted_offer = None

        # Try finding a "Hook" price first (e.g. "From ₹999")
        hook_match = hook_price_regex.search(cleaned_text)
        loose_match = loose_price_regex.search(cleaned_text)
        offer_match = offer_regex.search(cleaned_text)
        
        if hook_match:
            callout_type = "Price Hook"
            extracted_price = hook_match.group(1)
        elif loose_match:
            # Fallback: We found a price, but no "From/Just" word
            callout_type = "Price Only"
            extracted_price = loose_match.group(1)
        
        if offer_match:
            if "Price" in callout_type:
                callout_type = "Price + Offer"
            else:
                callout_type = "Offer"
            extracted_offer = offer_match.group(1)

        # --- CLEAN UP: Fix common OCR errors ---
        # If we extracted something like "?29,999", replace the ? with ₹
        if extracted_price:
            extracted_price = extracted_price.replace("?", "₹")

        return {
            "has_face": has_face,
            "brightness_level": brightness_level,
            "callout_type": callout_type,
            "extracted_price": extracted_price, 
            "extracted_offer": extracted_offer,
            "raw_text": raw_text # Keep raw text for debugging
        }
    except Exception as e:
        print(f"Error analyzing image: {e}") 
        return {"error": str(e)}

# --- 2. REPORTING UI ---

def display_aggregate_report(above_avg_df, below_avg_df, metric):
    st.markdown("--- \n ## 2. Performance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Above Average")
        st.caption(f"Top performers (> avg {metric})")
        if not above_avg_df.empty:
            # Group by Price Callout and Count
            top_prices = Counter(above_avg_df['extracted_price'].dropna()).most_common(5)
            if top_prices:
                df_prices = pd.DataFrame(top_prices, columns=["Price Callout", "Count"])
                st.dataframe(df_prices, use_container_width=True, hide_index=True)
            else:
                st.info("No prices extracted in this group.")
    
    with col2:
        st.markdown("### Below Average")
        st.caption(f"Low performers (<= avg {metric})")
        if not below_avg_df.empty:
            top_prices_below = Counter(below_avg_df['extracted_price'].dropna()).most_common(5)
            if top_prices_below:
                df_prices_below = pd.DataFrame(top_prices_below, columns=["Price Callout", "Count"])
                st.dataframe(df_prices_below, use_container_width=True, hide_index=True)
            else:
                st.info("No prices extracted in this group.")

def display_full_data(df_sorted, metric, image_name_col):
    st.markdown("--- \n ## 1. Detailed Data (Debug View)")
    st.markdown("Check the **'Raw Text'** column to see exactly what the AI read.")
    
    # Only show columns that actually exist in the dataframe
    possible_cols = [image_name_col, metric, 'extracted_price', 'extracted_offer', 'callout_type', 'raw_text', 'has_face']
    display_cols = [c for c in possible_cols if c in df_sorted.columns]
    
    st.dataframe(df_sorted[display_cols], use_container_width=True)

# --- 3. MAIN APP ---

st.set_page_config(layout="wide")
st.title("Creative Analysis Dashboard")

st.sidebar.header("1. Upload Files")
csv_file = st.sidebar.file_uploader("Upload Metrics CSV", type="csv")
uploaded_images = st.sidebar.file_uploader("Upload Creative Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.sidebar.header("2. Configure Columns")
metric_col = st.sidebar.text_input("Metric Column (e.g. CTR)", "CTR")
image_name_col = st.sidebar.text_input("Image Name Column", "image_name")

if st.sidebar.button("Run Analysis", use_container_width=True):
    if csv_file and uploaded_images:
        with st.spinner("Loading AI Models..."):
            face_cascade, ocr_reader = load_models()
            if not face_cascade: st.stop()

        # Read CSV
        try:
            df = pd.read_csv(csv_file)
            # Validate columns
            if metric_col not in df.columns or image_name_col not in df.columns:
                st.error(f"Columns not found! CSV has: {list(df.columns)}")
                st.stop()
        except Exception as e:
            st.error(f"CSV Error: {e}")
            st.stop()

        # Map images
        images_dict = {f.name: f.getvalue() for f in uploaded_images}
        
        # Run Loop
        all_features = []
        bar = st.progress(0, text="Analyzing...")
        total_rows = len(df)
        
        for i, row in df.iterrows():
            img_name = row[image_name_col]
            if img_name in images_dict:
                feats = analyze_image_features(images_dict[img_name], face_cascade, ocr_reader)
                if "error" not in feats:
                    # Merge CSV row + AI features
                    all_features.append({**row.to_dict(), **feats})
            
            # Update progress bar
            bar.progress((i + 1) / total_rows)
            
        bar.empty()
        
        if not all_features:
            st.error("No matching images analyzed. Check your CSV filenames match your image uploads.")
            st.stop()

        # Build DataFrame
        res_df = pd.DataFrame(all_features)
        
        # Convert metric to numeric, forcing errors to NaN
        res_df[metric_col] = pd.to_numeric(res_df[metric_col], errors='coerce')
        res_df = res_df.sort_values(by=metric_col, ascending=False)
        
        # Display Results
        display_full_data(res_df, metric_col, image_name_col)
        
        mean_val = res_df[metric_col].mean()
        st.markdown(f"*(Average {metric_col}: {mean_val:.4f})*")
        
        display_aggregate_report(
            res_df[res_df[metric_col] > mean_val], 
            res_df[res_df[metric_col] <= mean_val], 
            metric_col
        )
    else:
        st.warning("Please upload both CSV and Images.")
