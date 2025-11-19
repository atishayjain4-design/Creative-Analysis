import streamlit as st
import pandas as pd
import cv2
import numpy as np
import re
import os
import easyocr
from io import BytesIO
from collections import Counter # New import for counting phrases

# --- 1. FEATURE EXTRACTION FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads OpenCV and EasyOCR models into memory."""
    print("Loading models...")
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
        contrast = np.std(gray)

        # --- OCR Features ---
        results = ocr_reader.readtext(image_bytes, detail=0, paragraph=True)
        extracted_text = " ".join(results).upper() # Convert to uppercase for easier matching
        
        has_text = len(extracted_text.strip()) > 5
        
        # --- NEW: Upgraded Callout Extraction ---
        # Regex with CAPTURING GROUPS (the parentheses) to extract the *actual* text.
        
        # This regex looks for things like "FROM ₹29,999", "$19.99", "ONLY ₹300"
        price_regex = re.compile(r"((?:FROM|STARTS AT|STARTING AT|ONLY)\s*(?:₹|\$|€|£)\s*[\d,.]+|(?:\$|€|£|₹)\s*[\d,.]+)")
        
        # This regex looks for "50% OFF", "SALE", "FREE SHIPPING", etc.
        offer_regex = re.compile(r"(\d{1,2}\s?% (?:OFF)?|SALE|FREE SHIPPING|FREE|BOGO|DEAL|OFFER)")
        
        callout_type = "None"
        extracted_price = None
        extracted_offer = None

        price_match = price_regex.search(extracted_text)
        offer_match = offer_regex.search(extracted_text)
        
        if price_match:
            callout_type = "Price Point"
            extracted_price = price_match.group(1) # Get the text from the first capturing group
        
        # An ad can have both a price AND an offer (e.g., "$29.99 - 50% OFF")
        if offer_match:
            # If it already had a price, just add the offer.
            # If not, set the callout type.
            if callout_type == "Price Point":
                callout_type = "Price + Offer"
            else:
                callout_type = "Offer"
            extracted_offer = offer_match.group(1)

        return {
            "has_face": has_face,
            "brightness": brightness,
            "brightness_level": brightness_level,
            "contrast": contrast,
            "has_text": has_text,
            "callout_type": callout_type,
            "extracted_price": extracted_price, # NEW
            "extracted_offer": extracted_offer, # NEW
            "extracted_text_snippet": extracted_text.strip()[:75] + "..." if has_text else "N/A"
        }
    except Exception as e:
        print(f"Error analyzing image: {e}") 
        return {"error": str(e)}

# --- 2. REPORTING FUNCTIONS (UPGRADED) ---

def display_aggregate_report(above_avg_df, below_avg_df, metric):
    st.markdown("--- \n ## 2. Aggregate Analysis: High-Performers vs. Low-Performers")
    st.markdown(f"Comparing **{len(above_avg_df)}** high-performing creatives against **{len(below_avg_df)}** low-performing ones.")

    # --- 1. Brightness Level Distribution ---
    st.markdown("### Brightness Level Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Above Average")
        st.bar_chart(above_avg_df['brightness_level'].value_counts(normalize=True))
    with col2:
        st.markdown("Below Average")
        st.bar_chart(below_avg_df['brightness_level'].value_counts(normalize=True))
    
    # --- 2. Callout Type Distribution ---
    st.markdown("### Callout Type Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Above Average")
        # Re-order the index for a consistent chart
        callout_counts_above = above_avg_df['callout_type'].value_counts(normalize=True)
        st.bar_chart(callout_counts_above.reindex(["Price + Offer", "Price Point", "Offer", "None"]).fillna(0))
    with col2:
        st.markdown("Below Average")
        callout_counts_below = below_avg_df['callout_type'].value_counts(normalize=True)
        st.bar_chart(callout_counts_below.reindex(["Price + Offer", "Price Point", "Offer", "None"]).fillna(0))

    # --- 3. Other Features ---
    st.markdown("### Other Features (as % of group)")
    bool_data = {
        "Above Avg (%)": {
            "Has Face": above_avg_df['has_face'].mean() * 100,
            "Has Text": above_avg_df['has_text'].mean() * 100
        },
        "Below Avg (%)": {
            "Has Face": below_avg_df['has_face'].mean() * 100,
            "Has Text": below_avg_df['has_text'].mean() * 100
        }
    }
    st.dataframe(pd.DataFrame(bool_data).T.style.format("{:.1f}%"))
    
    # --- 4. NEW: Top Extracted Callouts (The "Why") ---
    st.markdown("### Top Callouts (The \"Why\")")
    st.markdown("This shows the *exact* callouts that appeared most often in each group.")
    
    # Helper to create a frequency DataFrame
    def get_top_phrases(series):
        # Count non-null values
        counts = Counter(series.dropna()).most_common(5)
        if not counts:
            return pd.DataFrame(columns=["Phrase", "Count"])
        return pd.DataFrame(counts, columns=["Phrase", "Count"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top Prices (Above Avg)")
        top_prices_above = get_top_phrases(above_avg_df['extracted_price'])
        st.dataframe(top_prices_above, use_container_width=True)
        
        st.markdown("#### Top Offers (Above Avg)")
        top_offers_above = get_top_phrases(above_avg_df['extracted_offer'])
        st.dataframe(top_offers_above, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Prices (Below Avg)")
        top_prices_below = get_top_phrases(below_avg_df['extracted_price'])
        st.dataframe(top_prices_below, use_container_width=True)
        
        st.markdown("#### Top Offers (Below Avg)")
        top_offers_below = get_top_phrases(below_avg_df['extracted_offer'])
        st.dataframe(top_offers_below, use_container_width=True)


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
        # Display new extracted fields
        st.write(f"**Callout Type:** {best.callout_type}")
        st.write(f"**Extracted Price:** {best.extracted_price or 'N/A'}")
        st.write(f"**Extracted Offer:** {best.extracted_offer or 'N/A'}")
        st.write(f"**Face:** {best.has_face}")
        st.write(f"**Brightness:** {best.brightness_level}")

    with col2:
        st.markdown("### 🥉 Worst Creative")
        st.markdown(f"**{worst['image_name']}**")
        st.markdown(f"**{metric}: {worst[metric]:.4f}**")
        if worst['image_name'] in images_dict:
            st.image(images_dict[worst['image_name']], use_column_width=True)
        # Display new extracted fields
        st.write(f"**Callout Type:** {worst.callout_type}")
        st.write(f"**Extracted Price:** {worst.extracted_price or 'N/A'}")
        st.write(f"**Extracted Offer:** {worst.extracted_offer or 'N/A'}")
        st.write(f"**Face:** {worst.has_face}")
        st.write(f"**Brightness:** {worst.brightness_level}")

# --- 3. MAIN APPLICATION UI (Mostly Unchanged) ---

st.set_page_config(layout="wide")
st.title("Creative Analysis Dashboard")
st.write("Upload your metrics and image folder to get an automated performance analysis. This runs 100% on Python.")

# --- Sidebar for Uploads and Config ---
st.sidebar.header("1. Upload Files")
csv_file = st.sidebar.file_uploader("Upload your Metrics CSV", type="csv")
uploaded_images = st.sidebar.file_uploader("Upload all your Creative Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.sidebar.header("2. Configure Columns")
metric_col = st.sidebar.text_input("Metric to Analyze (from CSV)", "CTR")
image_name_col = st.sidebar.text_input("Image Filename Column (from CSV)", "image_name")

if st.sidebar.button("Run Analysis", use_container_width=True):
    if csv_file is not None and len(uploaded_images) > 0:
        
        # --- Run Analysis ---
        with st.spinner("Initializing models... (This happens once)"):
            face_cascade, ocr_reader = load_models()
        
        if face_cascade is None or ocr_reader is None:
            st.stop() 

        with st.spinner("Parsing CSV..."):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                st.stop()
        
        # Check for columns
        if image_name_col not in df.columns:
            st.error(f"Error: Image column '{image_name_col}' not found in CSV. Found columns: {list(df.columns)}")
            st.stop()
        if metric_col not in df.columns:
            st.error(f"Error: Metric column '{metric_col}' not found in CSV. Found columns: {list(df.columns)}")
            st.stop()

        images_dict = {f.name: f.getvalue() for f in uploaded_images}
        
        all_features = []
        progress_bar = st.progress(0, text="Analyzing creatives...")
        total_images = len(df)
        
        for index, row in df.iterrows():
            image_name = row[image_name_col]
            
            if image_name in images_dict:
                image_bytes = images_dict[image_name]
                features = analyze_image_features(image_bytes, face_cascade, ocr_reader)
                
                if "error" not in features:
                    combined_data = {**row, **features}
                    all_features.append(combined_data)
            else:
                print(f"Skipping row {index}: Image '{image_name}' not found in uploaded files.") 
            
            progress_bar.progress((index + 1) / total_images, text=f"Analyzing {image_name} ({index + 1}/{total_images})")

        progress_bar.empty()

        if not all_features:
            st.error("Analysis failed. No matching images found between the CSV and the uploaded files. Check your `image_name_col` and file names.")
        else:
            # --- Create the Full DataFrame ---
            df_with_features = pd.DataFrame(all_features)
            
            try:
                df_with_features[metric_col] = pd.to_numeric(df_with_features[metric_col])
            except ValueError:
                st.error(f"Error: The metric column '{metric_col}' contains non-numeric values. Please clean your CSV.")
                st.stop()
            
            df_sorted = df_with_features.sort_values(by=metric_col, ascending=False)
            
            # --- Display Reports ---
            st.markdown("## 1. Complete Analysis Data Table")
            
            # Define columns to show, including new extracted ones
            display_cols = [
                image_name_col, 
                metric_col, 
                'callout_type', 
                'extracted_price', # NEW
                'extracted_offer', # NEW
                'has_face', 
                'brightness_level', 
                'extracted_text_snippet'
            ]
            st.dataframe(df_sorted[display_cols])

            # Groups for aggregate analysis
            mean_metric = df_sorted[metric_col].mean()
            st.markdown(f"*(The average **{metric_col}** for this dataset is **{mean_metric:.4f}**)*")
            
            above_avg_df = df_sorted[df_sorted[metric_col] > mean_metric]
            below_avg_df = df_sorted[df_sorted[metric_col] <= mean_metric]

            if above_avg_df.empty or below_avg_df.empty:
                st.warning("Cannot perform aggregate analysis: all creatives are on one side of the average.")
            else:
                display_aggregate_report(above_avg_df, below_avg_df, metric_col)
                
                best_worst_images = {}
                if not df_sorted.empty:
                    best_name = df_sorted.iloc[0]['image_name']
                    worst_name = df_sorted.iloc[-1]['image_name']
                    if best_name in images_dict:
                        best_worst_images[best_name] = images_dict[best_name]
                    if worst_name in images_dict:
                        best_worst_images[worst_name] = images_dict[worst_name]
                
                display_best_vs_worst(df_sorted, metric_col, best_worst_images)

    else:
        st.warning("Please upload a CSV file and at least one image.")
