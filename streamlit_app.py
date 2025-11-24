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
from PIL import Image
# --- NEW: Import the AI library ---
from rembg import remove 

# --- 1. FEATURE EXTRACTION FUNCTIONS ---

@st.cache_resource
def load_models():
    """Loads OpenCV and EasyOCR models into memory."""
    print("Loading models...")
    
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
    
    print("Loading EasyOCR model...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    print("Models loaded.")
    return face_cascade, ocr_reader

def analyze_image_features(image_bytes, face_cascade, ocr_reader):
    try:
        # --- PRE-PROCESSING ---
        # 1. Load as PIL Image (Required for rembg AI)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # 2. Load as OpenCV Image (Required for Face/Text/Brightness)
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_cv is None: return {"error": "Decode failed"}
        
        # Convert OpenCV to Grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        height, width, _ = image_cv.shape
        total_pixels = width * height

        # --- A. AI PRODUCT DETECTION (rembg) ---
        # This runs the U-2-Net Neural Network to separate product from background.
        # It solves the "Ghost" (white on white) and "Zebra" (stripes) problems.
        output_image = remove(pil_image)
        output_np = np.array(output_image)
        
        # The Alpha channel (4th layer) is our perfect product mask
        # 0 = Background, 255 = Product
        if output_np.shape[2] == 4:
            alpha_mask = output_np[:, :, 3]
        else:
            # Fallback if image has no alpha (rare with rembg)
            alpha_mask = np.zeros_like(gray)

        # 1. Exact Product Size
        product_pixels = cv2.countNonZero(alpha_mask)
        product_area_pct = (product_pixels / total_pixels) * 100
        
        if product_area_pct < 15: product_size_bucket = "Small (<15%)"
        elif product_area_pct < 40: product_size_bucket = "Medium (15-40%)"
        else: product_size_bucket = "Large (>40%)"

        # 2. Perfect Contrast & Background Analysis
        if product_pixels == 0:
            contrast_val = 0
            bg_label = "Unknown"
        else:
            # Mean brightness of Product (Use AI mask)
            prod_brightness = cv2.mean(gray, mask=alpha_mask)[0]
            
            # Mean brightness of Background (Invert AI mask)
            bg_mask = cv2.bitwise_not(alpha_mask)
            bg_brightness = cv2.mean(gray, mask=bg_mask)[0]
            
            contrast_val = abs(prod_brightness - bg_brightness)
            
            if bg_brightness < 90: bg_label = "Dark Background"
            elif bg_brightness < 170: bg_label = "Medium Background"
            else: bg_label = "Light/White Background"
            
        # Label Contrast
        if contrast_val < 40: contrast_label = "Low Contrast"
        elif contrast_val < 90: contrast_label = "Medium Contrast"
        else: contrast_label = "High Contrast"

        # --- B. FACE DETECTION ---
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        has_face = len(faces) > 0
        
        # Overall Brightness
        brightness = np.mean(gray)
        if brightness < 90: brightness_level = "Low (Dark)"
        elif brightness < 180: brightness_level = "Medium (Balanced)"
        else: brightness_level = "High (Bright)"

        # --- C. VISUAL STYLE ---
        texture_score = np.std(gray)
        if texture_score < 40: visual_style = "2D / Flat"
        elif texture_score < 60: visual_style = "Mixed"
        else: visual_style = "3D / Photo"

        # --- D. OCR & TEXT ANALYSIS ---
        ocr_results = ocr_reader.readtext(image_bytes, detail=1, paragraph=False)
        
        all_text_parts = []
        max_font_height = 0
        max_text_length = 0
        headline_text = "None"
        main_body_text = "None"
        
        for (bbox, text, prob) in ocr_results:
            all_text_parts.append(text)
            # Headline (Largest Font)
            box_height = bbox[2][1] - bbox[0][1]
            if box_height > max_font_height:
                max_font_height = box_height
                headline_text = text
            # Main Text (Longest Block > 3 chars)
            if len(text) > max_text_length and len(text) > 3:
                max_text_length = len(text)
                main_body_text = text

        raw_text = " ".join(all_text_parts)
        cleaned_text = raw_text.upper()
        
        # --- E. PRICE EXTRACTION ---
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
            "headline_text": headline_text,
            "main_body_text": main_body_text,
            "product_area_pct": product_area_pct,
            "product_size_bucket": product_size_bucket,
            "bg_label": bg_label,
            "contrast_label": contrast_label,
            "visual_style": visual_style,
            "has_face": has_face,
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
        'bg_label', 'contrast_label',
        'product_area_pct', 'main_body_text', 'headline_text',
        'extracted_price', 'extracted_offer', 
        'has_face', 'visual_style'
    ]
    cols = [c for c in cols if c in df_sorted.columns]
    
    display_df = df_sorted[cols].copy()
    if 'product_area_pct' in display_df.columns:
        display_df['product_area_pct'] = display_df['product_area_pct'].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True)

def display_aggregate_report(above_bench_df, below_bench_df, metric, benchmark):
    st.markdown("--- \n ## 2. Aggregate Analysis")
    st.markdown(f"Comparing **{len(above_bench_df)}** Top Performers (> {benchmark}) vs. **{len(below_bench_df)}** Low Performers (<= {benchmark}).")

    # --- Background & Contrast ---
    st.markdown("### Background & Contrast")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Performers (BG):**")
        if not above_bench_df.empty:
            st.bar_chart(above_bench_df['bg_label'].value_counts(normalize=True))
    with col2:
        st.markdown("**Top Performers (Contrast):**")
        if not above_bench_df.empty:
            st.bar_chart(above_bench_df['contrast_label'].value_counts(normalize=True))

    st.divider()

    # --- Visual Stats ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Visual Style")
        if not above_bench_df.empty:
            st.markdown("**Top Performers:**")
            st.bar_chart(above_bench_df['visual_style'].value_counts(normalize=True))
            
    with col2:
        st.markdown("### Product Image Size")
        if not above_bench_df.empty:
            st.markdown("**Top Performers:**")
            st.bar_chart(above_bench_df['product_size_bucket'].value_counts(normalize=True))

    # --- Face Detection ---
    st.markdown("### Human Element (Has Face?)")
    col1, col2 = st.columns(2)
    with col1:
        if not above_bench_df.empty:
            st.markdown(f"**Above {benchmark}**")
            face_counts = above_bench_df['has_face'].astype(str).value_counts(normalize=True)
            st.bar_chart(face_counts.reindex(['True', 'False']).fillna(0))
    with col2:
        if not below_bench_df.empty:
            st.markdown(f"**Below {benchmark}**")
            face_counts_below = below_bench_df['has_face'].astype(str).value_counts(normalize=True)
            st.bar_chart(face_counts_below.reindex(['True', 'False']).fillna(0))
    
    # --- Top Callouts ---
    st.markdown("### Top Callouts")
    
    def get_top_phrases(series):
        counts = Counter(series.dropna()).most_common(5)
        return pd.DataFrame(counts, columns=["Phrase", "Count"]) if counts else None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top Prices (> {benchmark})**")
        df_p = get_top_phrases(above_bench_df['extracted_price'])
        if df_p is not None: st.dataframe(df_p, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown(f"**Top Offers (> {benchmark})**")
        df_o = get_top_phrases(above_bench_df['extracted_offer'])
        if df_o is not None: st.dataframe(df_o, use_container_width=True, hide_index=True)


def display_best_vs_worst(df_sorted, metric, images_dict):
    st.markdown("--- \n ## 3. Best vs. Worst (Overall)")
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
            
            st.success(f"**Headline:** {item.get('headline_text', 'N/A')}")
            st.info(f"**Main Text:** {item.get('main_body_text', 'N/A')}")
            st.info(f"**Background:** {item.get('bg_label', '-')}")
            st.info(f"**Product Coverage:** {item.get('product_area_pct', 0):.1f}%")
            st.write(f"**Face:** {'Yes' if item.has_face else 'No'}")
            st.write(f"**Price:** {item.extracted_price or '-'}")
            st.write(f"**Offer:** {item.extracted_offer or '-'}")

# --- 3. MAIN APP ---

st.set_page_config(layout="wide")
st.title("Creative Analysis Dashboard")

st.sidebar.header("1. Upload Files")
csv_file = st.sidebar.file_uploader("Upload Metrics CSV", type="csv")
uploaded_images = st.sidebar.file_uploader("Upload Creative Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

st.sidebar.header("2. Configure Analysis")
metric_col = st.sidebar.text_input("Metric Column (e.g. CTR)", "CTR")
image_name_col = st.sidebar.text_input("Image Name Column", "image_name")
benchmark_val = st.sidebar.number_input("Benchmark Value (Split High/Low)", value=1.1, step=0.1)

if st.sidebar.button("Run Analysis", use_container_width=True):
    if csv_file and uploaded_images:
        with st.spinner("Loading AI Models..."):
            face_cascade, ocr_reader = load_models()
            if not face_cascade: st.stop()

        # --- ROBUST CSV LOADING ---
        try:
            df = pd.read_csv(csv_file)
            if image_name_col in df.columns:
                df[image_name_col] = df[image_name_col].astype(str).str.strip()
            
            if metric_col not in df.columns or image_name_col not in df.columns:
                st.error(f"Columns not found! CSV has: {list(df.columns)}")
                st.stop()
        except Exception as e:
            st.error(f"CSV Error: {e}")
            st.stop()

        # --- ROBUST IMAGE MAPPING ---
        images_dict = {f.name.strip(): f.getvalue() for f in uploaded_images}
        
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
            with st.expander("Debug: Filename Mismatch Checker", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Filenames in CSV (First 5):**")
                    st.write(df[image_name_col].head(5).tolist())
                with col_b:
                    st.write("**Filenames Uploaded (First 5):**")
                    st.write(list(images_dict.keys())[:5])
            st.stop()

        res_df = pd.DataFrame(all_features)
        res_df[metric_col] = pd.to_numeric(res_df[metric_col], errors='coerce')
        res_df = res_df.sort_values(by=metric_col, ascending=False)
        
        display_full_data(res_df, metric_col, image_name_col)
        
        mean_val = res_df[metric_col].mean()
        col1, col2 = st.columns(2)
        col1.metric(label=f"Dataset Average {metric_col}", value=f"{mean_val:.4f}")
        col2.metric(label="Benchmark Used", value=f"{benchmark_val}")
        
        display_aggregate_report(
            res_df[res_df[metric_col] > benchmark_val], 
            res_df[res_df[metric_col] <= benchmark_val], 
            metric_col,
            benchmark_val
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
