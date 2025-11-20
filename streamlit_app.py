import streamlit as st
import pandas as pd
import cv2
import numpy as np
import re
import os
import easyocr
import urllib.request
from io import BytesIO
from collections import Counter

# --- 1. FEATURE EXTRACTION FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads OpenCV and EasyOCR models into memory."""
    print("Loading models...")
    
    # --- UPDATED: Using 'alt2' model which is much better at avoiding false positives ---
    cascade_filename = 'haarcascade_frontalface_alt2.xml'
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    
    # Check if model exists, if not, DOWNLOAD it automatically
    if not os.path.exists(cascade_filename):
        print(f"Model not found. Downloading {cascade_filename}...")
        try:
            urllib.request.urlretrieve(cascade_url, cascade_filename)
            print("Download complete.")
        except Exception as e:
            st.error(f"Fatal Error: Could not download face detection model. Check internet connection. Error: {e}")
            return None, None
            
    # Load the classifier
    face_cascade = cv2.CascadeClassifier(cascade_filename)
    if face_cascade.empty():
        st.error("Fatal Error: Loaded XML file is empty or corrupted.")
        return None, None
    
    # 2. Setup OCR Model
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

        if image_cv is None:
            return {"error": "Could not decode image"}

        # --- CV Features ---
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # --- UPDATED FACE DETECTION PARAMETERS ---
        # scaleFactor=1.1: Checks image at different sizes (standard)
        # minNeighbors=5:  Stricter threshold. Requires 5 detections in one spot to count as a face.
        #                  (Previous was 4. Increasing this eliminates random noise/false positives).
        # minSize=(50, 50): Ignores very small blurry patches that often look like faces.
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        has_face = len(faces) > 0
        
        # Calculate Brightness
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
        
        # --- ULTRA-ROBUST PRICE EXTRACTION LOGIC ---
        
        # 1. Permissive Hook Regex
        hook_price_regex = re.compile(r"((?:FROM|STARTS?|STARTING|JUST|ONLY|NOW|AT|@)\s*(?:[^0-9\s]{0,3})\s*[\d,.]+(?:/-)?)")

        # 2. Loose Price Regex
        loose_price_regex = re.compile(r"((?:₹|\$|€|£|RS\.?|INR|\?)\s*[\d,.]+(?:/-)?)")
        
        # 3. Suffix Price Regex
        suffix_price_regex = re.compile(r"([\d,.]+/-)")
        
        # 4. Offer Regex
        offer_regex = re.compile(r"(\d{1,2}\s?% (?:OFF)?|SALE|FREE SHIPPING|FREE|BOGO|DEAL|OFFER|FLAT \d+%)")
        
        callout_type = "None"
        extracted_price = None
        extracted_offer = None

        # Attempt matches in order of specificity
        hook_match = hook_price_regex.search(cleaned_text)
        loose_match = loose_price_regex.search(cleaned_text)
        suffix_match = suffix_price_regex.search(cleaned_text)
        offer_match = offer_regex.search(cleaned_text)
        
        if hook_match:
            callout_type = "Price Hook"
            extracted_price = hook_match.group(1)
        elif loose_match:
            callout_type = "Price Only"
            extracted_price = loose_match.group(1)
        elif suffix_match:
            callout_type = "Price Only"
            extracted_price = suffix_match.group(1)
        
        if offer_match:
            if "Price" in callout_type:
                callout_type = "Price + Offer"
            else:
                callout_type = "Offer"
            extracted_offer = offer_match.group(1)

        # --- CLEAN UP: Fix common OCR errors ---
        if extracted_price:
            extracted_price = extracted_price.replace("?", "₹").replace("~", "₹")
            extracted_price = re.sub(r"([A-Z])(₹)", r"\1 \2", extracted_price)

        return {
            "has_face": has_face,
            "face_count": len(faces),
            "brightness_level": brightness_level,
            "callout_type": callout_type,
            "extracted_price": extracted_price, 
            "extracted_offer": extracted_offer,
            "raw_text": raw_text
        }
    except Exception as e:
        print(f"Error analyzing image: {e}") 
        return {"error": str(e)}

# --- 2. REPORTING FUNCTIONS ---

def display_full_data(df_sorted, metric, image_name_col):
    """Displays the detailed data table for debugging."""
    st.markdown("--- \n ## 1. Detailed Data (Debug View)")
    st.markdown("Check the columns to see what the AI extracted.")
    
    cols = [image_name_col, metric, 'has_face', 'extracted_price', 'extracted_offer', 'callout_type', 'raw_text']
    cols = [c for c in cols if c in df_sorted.columns]
    
    st.dataframe(df_sorted[cols], use_container_width=True)

def display_aggregate_report(above_avg_df, below_avg_df, metric):
    st.markdown("--- \n ## 2. Aggregate Analysis: High-Performers vs. Low-Performers")
    st.markdown(f"Comparing **{len(above_avg_df)}** high-performing creatives against **{len(below_avg_df)}** low-performing ones.")

    # --- 1. Face Detection Distribution ---
    st.markdown("### Human Element (Has Face?)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Above Average**")
        if not above_avg_df.empty:
            face_counts = above_avg_df['has_face'].astype(str).value_counts(normalize=True)
            face_counts = face_counts.reindex(['True', 'False']).fillna(0)
            st.bar_chart(face_counts)
    with col2:
        st.markdown("**Below Average**")
        if not below_avg_df.empty:
            face_counts_below = below_avg_df['has_face'].astype(str).value_counts(normalize=True)
            face_counts_below = face_counts_below.reindex(['True', 'False']).fillna(0)
            st.bar_chart(face_counts_below)

    # --- 2. Brightness Level Distribution ---
    st.markdown("### Brightness Level Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Above Average")
        if not above_avg_df.empty:
            st.bar_chart(above_avg_df['brightness_level'].value_counts(normalize=True))
    with col2:
        st.markdown("Below Average")
        if not below_avg_df.empty:
            st.bar_chart(below_avg_df['brightness_level'].value_counts(normalize=True))
    
    # --- 3. Callout Type Distribution ---
    st.markdown("### Callout Type Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Above Average")
        if not above_avg_df.empty:
            callout_counts_above = above_avg_df['callout_type'].value_counts(normalize=True)
            st.bar_chart(callout_counts_above.reindex(["Price + Offer", "Price Hook", "Price Only", "Offer", "None"]).fillna(0))
    with col2:
        st.markdown("Below Average")
        if not below_avg_df.empty:
            callout_counts_below = below_avg_df['callout_type'].value_counts(normalize=True)
            st.bar_chart(callout_counts_below.reindex(["Price + Offer", "Price Hook", "Price Only", "Offer", "None"]).fillna(0))
    
    # --- 4. Top Extracted Callouts ---
    st.markdown("### Top Callouts (The \"Why\")")
    st.markdown("These tables show exactly which prices and offers appeared most often.")
    
    def get_top_phrases(series):
        counts = Counter(series.dropna()).most_common(5)
        if not counts:
            return pd.DataFrame(columns=["Phrase", "Count"])
        return pd.DataFrame(counts, columns=["Phrase", "Count"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top Prices (Above Avg)")
        st.dataframe(get_top_phrases(above_avg_df['extracted_price']), use_container_width=True, hide_index=True)
        st.markdown("#### Top Offers (Above Avg)")
        st.dataframe(get_top_phrases(above_avg_df['extracted_offer']), use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Top Prices (Below Avg)")
        st.dataframe(get_top_phrases(below_avg_df['extracted_price']), use_container_width=True, hide_index=True)
        st.markdown("#### Top Offers (Below Avg)")
        st.dataframe(get_top_phrases(below_avg_df['extracted_offer']), use_container_width=True, hide_index=True)


def display_best_vs_worst(df_sorted, metric, images_dict):
    st.markdown("--- \n ## 3. Case Study: Best Creative vs. Worst Creative")
    
    if len(df_sorted) == 0:
        st.warning("No data to display for best vs. worst.")
        return

    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🥇 Best Creative")
        st.markdown(f"**{best['image_name']}**")
        st.markdown(f"**{metric}: {best[metric]:.4f}**")
        if best['image_name'] in images_dict:
            st.image(images_dict[best['image_name']], use_column_width=True)
        
        st.write(f"**Face Detected:** {'✅ Yes' if best.has_face else '❌ No'}")
        st.write(f"**Extracted Price:** {best.extracted_price or 'N/A'}")
        st.write(f"**Extracted Offer:** {best.extracted_offer or 'N/A'}")
        st.write(f"**Callout Type:** {best.callout_type}")
        st.write(f"**Brightness:** {best.brightness_level}")

    with col2:
        st.markdown("### 🥉 Worst Creative")
        st.markdown(f"**{worst['image_name']}**")
        st.markdown(f"**{metric}: {worst[metric]:.4f}**")
        if worst['image_name'] in images_dict:
            st.image(images_dict[worst['image_name']], use_column_width=True)
        
        st.write(f"**Face Detected:** {'✅ Yes' if worst.has_face else '❌ No'}")
        st.write(f"**Extracted Price:** {worst.extracted_price or 'N/A'}")
        st.write(f"**Extracted Offer:** {worst.extracted_offer or 'N/A'}")
        st.write(f"**Callout Type:** {worst.callout_type}")
        st.write(f"**Brightness:** {worst.brightness_level}")

# --- 3. MAIN APPLICATION UI ---

st.set_page_config(layout="wide")
st.title("Creative Analysis Dashboard")

st.sidebar.header("1. Upload Files")
csv_file = st.sidebar.file_uploader("Upload Metrics CSV", type="csv")
uploaded_images = st.sidebar.file_uploader("Upload all your Creative Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.sidebar.header("2. Configure Columns")
metric_col = st.sidebar.text_input("Metric Column (e.g. CTR)", "CTR")
image_name_col = st.sidebar.text_input("Image Name Column", "image_name")

if st.sidebar.button("Run Analysis", use_container_width=True):
    if csv_file and uploaded_images:
        with st.spinner("Loading AI Models..."):
            face_cascade, ocr_reader = load_models()
            if not face_cascade: st.stop()

        try:
            df = pd.read_csv(csv_file)
            # Validate columns
            if metric_col not in df.columns or image_name_col not in df.columns:
                st.error(f"Columns not found! CSV has: {list(df.columns)}")
                st.stop()
        except Exception as e:
            st.error(f"CSV Error: {e}")
            st.stop()

        images_dict = {f.name: f.getvalue() for f in uploaded_images}
        
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
        
        best_worst_images = {}
        if not res_df.empty:
            best_name = res_df.iloc[0][image_name_col]
            worst_name = res_df.iloc[-1][image_name_col]
            if best_name in images_dict:
                best_worst_images[best_name] = images_dict[best_name]
            if worst_name in images_dict:
                best_worst_images[worst_name] = images_dict[worst_name]
        
        display_best_vs_worst(res_df, metric_col, best_worst_images)
        
    else:
        st.warning("Please upload both CSV and Images.")
