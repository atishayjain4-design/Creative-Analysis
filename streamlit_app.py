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
    
    # Use 'alt2' model for better face detection
    cascade_filename = 'haarcascade_frontalface_alt2.xml'
    cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    
    if not os.path.exists(cascade_filename):
        print(f"Model not found. Downloading {cascade_filename}...")
        try:
            urllib.request.urlretrieve(cascade_url, cascade_filename)
        except Exception as e:
            st.error(f"Fatal Error: Could not download face model. {e}")
            return None, None
            
    face_cascade = cv2.CascadeClassifier(cascade_filename)
    if face_cascade.empty():
        st.error("Fatal Error: Loaded XML file is empty.")
        return None, None
    
    print("Loading EasyOCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    print("Models loaded.")
    return face_cascade, ocr_reader

def analyze_image_features(image_bytes, face_cascade, ocr_reader):
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_cv is None: return {"error": "Decode failed"}

        height, width, _ = image_cv.shape
        total_area = width * height

        # --- 1. CV Features ---
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Face Detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        has_face = len(faces) > 0
        
        # Brightness
        brightness = np.mean(gray)
        if brightness < 90: brightness_level = "Low (Dark)"
        elif brightness < 180: brightness_level = "Medium (Balanced)"
        else: brightness_level = "High (Bright)"

        # --- 2. 2D vs 3D / Depth Analysis ---
        # We use the standard deviation of pixel intensities as a proxy for "texture"
        # High std dev = lots of variation (shadows, texture, photos) = 3D/Depth
        # Low std dev = flat colors, uniform areas = 2D/Flat
        texture_score = np.std(gray)
        if texture_score < 40:
            visual_style = "2D / Flat"
        elif texture_score < 60:
            visual_style = "Mixed"
        else:
            visual_style = "3D / Photo / Depth"

        # --- 3. Product Image Size Estimation ---
        # We try to find the largest object that isn't the whole image frame
        # Blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        product_area_pct = 0.0
        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Take the largest contour that isn't the *entire* image (frame)
            for c in contours:
                area = cv2.contourArea(c)
                # Ignore if it covers > 95% (likely a border) or < 5% (noise)
                if 0.05 * total_area < area < 0.95 * total_area:
                    product_area_pct = (area / total_area) * 100
                    break
        
        # Format the size
        if product_area_pct < 15: product_size_label = "Small (<15%)"
        elif product_area_pct < 40: product_size_label = "Medium (15-40%)"
        else: product_size_label = "Large (>40%)"


        # --- 4. Advanced OCR (Text & Layout) ---
        # detail=1 gives us bounding box coordinates: [[x,y], [x,y], [x,y], [x,y]]
        ocr_results = ocr_reader.readtext(image_bytes, detail=1, paragraph=False)
        
        all_text_parts = []
        max_font_height = 0
        headline_text = "None"
        
        for (bbox, text, prob) in ocr_results:
            all_text_parts.append(text)
            
            # Calculate font height (y_bottom - y_top)
            # bbox is [[tl], [tr], [br], [bl]]
            box_height = bbox[2][1] - bbox[0][1]
            
            if box_height > max_font_height:
                max_font_height = box_height
                headline_text = text

        raw_text = " ".join(all_text_parts)
        cleaned_text = raw_text.upper()

        # --- Price Extraction ---
        hook_price_regex = re.compile(r"((?:FROM|STARTS?|STARTING|JUST|ONLY|NOW|AT|@)\s*(?:[^0-9\s]{0,3})\s*[\d,.]+(?:/-)?)")
        loose_price_regex = re.compile(r"((?:₹|\$|€|£|RS\.?|INR|\?)\s*[\d,.]+(?:/-)?)")
        suffix_price_regex = re.compile(r"([\d,.]+/-)")
        offer_regex = re.compile(r"(\d{1,2}\s?% (?:OFF)?|SALE|FREE SHIPPING|FREE|BOGO|DEAL|OFFER|FLAT \d+%)")
        
        callout_type = "None"
        extracted_price = None
        extracted_offer = None

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
            if "Price" in callout_type: callout_type = "Price + Offer"
            else: callout_type = "Offer"
            extracted_offer = offer_match.group(1)

        if extracted_price:
            extracted_price = extracted_price.replace("?", "₹").replace("~", "₹")
            extracted_price = re.sub(r"([A-Z])(₹)", r"\1 \2", extracted_price)

        return {
            "has_face": has_face,
            "visual_style": visual_style,         # NEW: 2D vs 3D
            "product_size_label": product_size_label, # NEW: Small/Med/Large
            "headline_text": headline_text,       # NEW: Text with largest font
            "brightness_level": brightness_level,
            "callout_type": callout_type,
            "extracted_price": extracted_price, 
            "extracted_offer": extracted_offer,
            "raw_text": raw_text
        }
    except Exception as e:
        print(f"Error: {e}") 
        return {"error": str(e)}

# --- 2. REPORTING UI ---

def display_full_data(df_sorted, metric, image_name_col):
    st.markdown("--- \n ## 1. Detailed Data")
    
    cols = [
        image_name_col, metric, 
        'visual_style', 'product_size_label', 
        'headline_text', 'extracted_price', 'extracted_offer', 
        'has_face'
    ]
    cols = [c for c in cols if c in df_sorted.columns]
    
    st.dataframe(df_sorted[cols], use_container_width=True)

def display_aggregate_report(above_avg_df, below_avg_df, metric):
    st.markdown("--- \n ## 2. Aggregate Analysis")
    st.markdown(f"Comparing **{len(above_avg_df)}** Top Performers vs. **{len(below_avg_df)}** Low Performers.")

    # --- Visual Style & Size ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Visual Style (2D vs 3D)")
        st.caption("Is the creative flat/graphical or photographic/depth?")
        if not above_avg_df.empty:
            st.markdown("**Above Average:**")
            st.bar_chart(above_avg_df['visual_style'].value_counts(normalize=True))
    with col2:
        st.markdown("### Product Image Size")
        st.caption("How large is the main object?")
        if not above_avg_df.empty:
            st.markdown("**Above Average:**")
            st.bar_chart(above_avg_df['product_size_label'].value_counts(normalize=True))

    st.divider()

    # --- Face Detection ---
    st.markdown("### Human Element (Has Face?)")
    col1, col2 = st.columns(2)
    with col1:
        if not above_avg_df.empty:
            face_counts = above_avg_df['has_face'].astype(str).value_counts(normalize=True)
            st.bar_chart(face_counts.reindex(['True', 'False']).fillna(0))
    with col2:
        if not below_avg_df.empty:
            face_counts_below = below_avg_df['has_face'].astype(str).value_counts(normalize=True)
            st.bar_chart(face_counts_below.reindex(['True', 'False']).fillna(0))
    
    # --- Top Callouts ---
    st.markdown("### Top Callouts")
    
    def get_top_phrases(series):
        counts = Counter(series.dropna()).most_common(5)
        return pd.DataFrame(counts, columns=["Phrase", "Count"]) if counts else None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Prices (Above Avg)**")
        df_p = get_top_phrases(above_avg_df['extracted_price'])
        if df_p is not None: st.dataframe(df_p, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown("**Top Offers (Above Avg)**")
        df_o = get_top_phrases(above_avg_df['extracted_offer'])
        if df_o is not None: st.dataframe(df_o, use_container_width=True, hide_index=True)


def display_best_vs_worst(df_sorted, metric, images_dict):
    st.markdown("--- \n ## 3. Best vs. Worst")
    
    if len(df_sorted) == 0: return

    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    for col, item, title in [(col1, best, "🥇 Best"), (col2, worst, "🥉 Worst")]:
        with col:
            st.markdown(f"### {title}")
            st.markdown(f"**{item['image_name']}**")
            st.markdown(f"**{metric}: {item[metric]:.4f}**")
            
            if item['image_name'] in images_dict:
                st.image(images_dict[item['image_name']], use_column_width=True)
            
            st.success(f"**Headline:** {item['headline_text']}")
            st.info(f"**Style:** {item['visual_style']}")
            st.write(f"**Product Size:** {item['product_size_label']}")
            st.write(f"**Face:** {'Yes' if item.has_face else 'No'}")
            st.write(f"**Price:** {item.extracted_price or '-'}")
            st.write(f"**Offer:** {item.extracted_offer or '-'}")

# --- 3. MAIN APP ---

st.set_page_config(layout="wide")
st.title("Creative Analysis Dashboard")

st.sidebar.header("1. Upload Files")
csv_file = st.sidebar.file_uploader("Upload Metrics CSV", type="csv")
uploaded_images = st.sidebar.file_uploader("Upload Creative Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

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
                    all_features.append({**row.to_dict(), **feats})
            bar.progress((i + 1) / total_rows)
        bar.empty()
        
        if not all_features:
            st.error("No matching images analyzed.")
            st.stop()

        res_df = pd.DataFrame(all_features)
        res_df[metric_col] = pd.to_numeric(res_df[metric_col], errors='coerce')
        res_df = res_df.sort_values(by=metric_col, ascending=False)
        
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
