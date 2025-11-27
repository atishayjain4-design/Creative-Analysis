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
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return face_cascade, ocr_reader

def analyze_image_features(image_bytes, face_cascade, ocr_reader):
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_cv is None: return {"error": "Decode Failed"}

        # Resize for stability
        h, w = image_cv.shape[:2]
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            image_cv = cv2.resize(image_cv, (int(w*scale), int(h*scale)))

        height, width, _ = image_cv.shape
        total_area = width * height

        # --- CV Features ---
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        has_face = len(faces) > 0
        
        brightness = np.mean(gray)
        if brightness < 90: brightness_level = "Low (Dark)"
        elif brightness < 180: brightness_level = "Medium (Balanced)"
        else: brightness_level = "High (Bright)"

        # --- Visual Style ---
        texture_score = np.std(gray)
        if texture_score < 40: visual_style = "2D / Flat"
        elif texture_score < 60: visual_style = "Mixed"
        else: visual_style = "3D / Photo"

        # --- PRODUCT DETECTION ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        roi_start_x = int(width * 0.40) 
        closed[:, :roi_start_x] = 0 
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        product_mask = np.zeros_like(gray)
        significant_area = 0.0
        
        if contours:
            for c in contours:
                area = cv2.contourArea(c)
                x, y, fw, fh = cv2.boundingRect(c)
                is_border = (fw > 0.95 * width) or (fh > 0.95 * height)
                if area > (0.02 * total_area) and not is_border:
                    significant_area += area
                    cv2.drawContours(product_mask, [c], -1, 255, -1)

        product_area_pct = min((significant_area / total_area) * 100, 100.0)
        if product_area_pct < 15: product_size_bucket = "Small (<15%)"
        elif product_area_pct < 40: product_size_bucket = "Medium (15-40%)"
        else: product_size_bucket = "Large (>40%)"

        # --- CONTRAST ---
        if cv2.countNonZero(product_mask) == 0:
            bg_brightness = np.mean(gray)
            contrast_val = 0
            bg_label = "Unknown"
        else:
            prod_brightness = cv2.mean(gray, mask=product_mask)[0]
            bg_mask = cv2.bitwise_not(product_mask)
            bg_brightness = cv2.mean(gray, mask=bg_mask)[0]
            contrast_val = abs(prod_brightness - bg_brightness)
            
            if bg_brightness < 90: bg_label = "Dark Background"
            elif bg_brightness < 170: bg_label = "Medium Background"
            else: bg_label = "Light/White Background"

        if contrast_val < 40: contrast_label = "Low Contrast"
        elif contrast_val < 90: contrast_label = "Medium Contrast"
        else: contrast_label = "High Contrast"

        # --- TEXT ANALYSIS ---
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        ocr_results = ocr_reader.readtext(image_rgb, detail=1, paragraph=False)
        
        text_blocks = []
        
        # Regex (With Monthly Support)
        price_suffix = r"(?:/-|/M|/MO|/MONTH|\s+PER\s+MONTH)?"
        hook_price_regex = re.compile(r"((?:FROM|STARTS?|STARTING|JUST|ONLY|NOW|AT|@)\s*(?:[^0-9\s]{0,3})\s*[\d,.]+" + price_suffix + r")")
        loose_price_regex = re.compile(r"((?:₹|\$|€|£|RS\.?|INR|\?)\s*[\d,.]+" + price_suffix + r")")
        offer_regex = re.compile(r"(\d{1,2}\s?% (?:OFF)?|SALE|FREE SHIPPING|FREE|BOGO|DEAL|OFFER|FLAT \d+%)")

        callout_y_top = float('inf')
        has_callout_block = False
        
        headline_text = "None"

        for result in ocr_results:
            bbox, text, conf = result
            text_clean = text.upper()
            
            tl, tr, br, bl = bbox
            box_width = tr[0] - tl[0]
            box_height = bl[1] - tl[1]
            box_area = box_width * box_height
            box_center_x = (tl[0] + tr[0]) / 2
            box_top_y = tl[1]
            box_bottom_y = bl[1]
            
            # Check Callout
            is_callout = False
            if hook_price_regex.search(text_clean) or loose_price_regex.search(text_clean) or offer_regex.search(text_clean):
                is_callout = True
                has_callout_block = True
                if box_top_y < callout_y_top:
                    callout_y_top = box_top_y

            text_blocks.append({
                'text': text,
                'h': box_height,
                'cx': box_center_x,
                'y_top': box_top_y,
                'y_bottom': box_bottom_y,
                'area': box_area,
                'is_callout': is_callout
            })

        # --- 1. CONSTRUCT HEADLINE (ALL TEXT) ---
        # We define "Headline" here as the aggregation of ALL significant text on the image
        # Filter: Height > 15px to ignore tiny noise
        significant_blocks = [b for b in text_blocks if b['h'] > 15]
        
        # Sort Top-to-Bottom so it reads naturally
        significant_blocks.sort(key=lambda x: x['y_top'])
        
        if significant_blocks:
            headline_text = " | ".join([b['text'] for b in significant_blocks])
        else:
            headline_text = "None"

        # --- 2. TITLE DETECTION ---
        title_text = "None"
        
        if has_callout_block:
            # Strategy A: Text strictly above the callout
            candidates = [b for b in text_blocks if b['y_bottom'] < callout_y_top and not b['is_callout']]
            candidates.sort(key=lambda x: x['y_bottom'], reverse=True)
            
            for cand in candidates:
                if cand['cx'] < (width * 0.75) and cand['h'] > 10: 
                    title_text = cand['text']
                    break
        
        if title_text == "None":
            # Strategy B: Largest Font on Left
            max_title_h = 0
            for b in text_blocks:
                is_left = b['cx'] < (width * 0.60)
                is_middle = (height * 0.10) < b['y_bottom'] < (height * 0.90)
                if is_left and is_middle and not b['is_callout']:
                    if b['h'] > max_title_h:
                        max_title_h = b['h']
                        title_text = b['text']

        # --- PRICE EXTRACTION ---
        # Use ALL detected text blocks for extraction, not just the headline, to ensure small offers are caught
        raw_text_full = " ".join([b['text'] for b in text_blocks])
        cleaned_text = raw_text_full.upper()
        
        callout_type = "None"
        extracted_price = None
        extracted_offer = None

        hook_match = hook_price_regex.search(cleaned_text)
        loose_match = loose_price_regex.search(cleaned_text)
        suffix_match = re.compile(r"([\d,.]+/-)").search(cleaned_text)
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
            "title_text": title_text,       
            "headline_text": headline_text, # Contains ALL significant text
            "bg_label": bg_label,
            "contrast_label": contrast_label,
            "product_area_pct": product_area_pct,
            "product_size_bucket": product_size_bucket,
            "visual_style": visual_style,
            "has_face": has_face,
            "brightness_level": brightness_level,
            "callout_type": callout_type,
            "extracted_price": extracted_price, 
            "extracted_offer": extracted_offer,
            "raw_text": raw_text_full
        }
    except Exception as e:
        return {"error": str(e)}

# --- 2. REPORTING UI ---

def display_full_data(df_sorted, metric, image_name_col):
    st.markdown("--- \n ## 3. Detailed Data")
    cols = [
        image_name_col, metric, 
        'title_text', 'headline_text', 
        'extracted_price', 'extracted_offer',
        'bg_label', 'contrast_label',
        'product_size_bucket', 'product_area_pct', 'has_face'
    ]
    cols = [c for c in cols if c in df_sorted.columns]
    
    display_df = df_sorted[cols].copy()
    if 'product_area_pct' in display_df.columns:
        display_df['product_area_pct'] = display_df['product_area_pct'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)

    st.dataframe(display_df, use_container_width=True)

def display_aggregate_report(above_bench_df, below_bench_df, metric, benchmark):
    st.markdown("--- \n ## 2. Aggregate Analysis")
    st.markdown(f"Comparing **{len(above_bench_df)}** Top Performers (> {benchmark}) vs. **{len(below_bench_df)}** Low Performers (<= {benchmark}).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Background")
        if not above_bench_df.empty:
            st.bar_chart(above_bench_df['bg_label'].value_counts(normalize=True))
    with col2:
        st.markdown("### Face Detection")
        if not above_bench_df.empty:
            face_counts = above_bench_df['has_face'].astype(str).value_counts(normalize=True)
            st.bar_chart(face_counts.reindex(['True', 'False']).fillna(0))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Contrast Level")
        if not above_bench_df.empty:
            st.caption(f"Above {benchmark}")
            st.bar_chart(above_bench_df['contrast_label'].value_counts(normalize=True))
        if not below_bench_df.empty:
            st.caption(f"Below {benchmark}")
            st.bar_chart(below_bench_df['contrast_label'].value_counts(normalize=True))

    with col2:
        st.markdown("### Product Size")
        if not above_bench_df.empty:
            st.caption(f"Above {benchmark}")
            st.bar_chart(above_bench_df['product_size_bucket'].value_counts(normalize=True))
        if not below_bench_df.empty:
            st.caption(f"Below {benchmark}")
            st.bar_chart(below_bench_df['product_size_bucket'].value_counts(normalize=True))
    
    st.markdown("### Top Text Elements")
    def get_top_phrases(series):
        counts = Counter(series.dropna()).most_common(5)
        return pd.DataFrame(counts, columns=["Phrase", "Count"]) if counts else None

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top Titles (Above Avg)**")
        df_t = get_top_phrases(above_bench_df['title_text'])
        if df_t is not None: st.dataframe(df_t, use_container_width=True, hide_index=True)
        st.markdown(f"**Top Prices (Above Avg)**")
        df_p = get_top_phrases(above_bench_df['extracted_price'])
        if df_p is not None: st.dataframe(df_p, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown(f"**Top Titles (Below Avg)**")
        df_tb = get_top_phrases(below_bench_df['title_text'])
        if df_tb is not None: st.dataframe(df_tb, use_container_width=True, hide_index=True)
        st.markdown(f"**Top Prices (Below Avg)**")
        df_pb = get_top_phrases(below_bench_df['extracted_price'])
        if df_pb is not None: st.dataframe(df_pb, use_container_width=True, hide_index=True)

def display_best_vs_worst(df_sorted, metric, images_dict):
    st.markdown("--- \n ## 1. Best vs. Worst (Overall)") 
    if len(df_sorted) == 0: return

    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    def get_img_bytes(name, img_dict, img_lookup):
        name_str = str(name).strip()
        if name_str in img_dict: return img_dict[name_str]
        if name_str.lower() in img_lookup: return img_dict[img_lookup[name_str.lower()]]
        return None

    img_lookup_display = {}
    for name in images_dict.keys():
        img_lookup_display[name.lower()] = name
        name_no_ext = os.path.splitext(name)[0]
        img_lookup_display[name_no_ext.lower()] = name

    for col, item, title in [(col1, best, "🥇 Best"), (col2, worst, "🥉 Worst")]:
        with col:
            st.markdown(f"### {title}")
            st.markdown(f"**{item[metric]:.4f}** ({metric})")
            
            img_name = item['image_name']
            img_bytes = get_img_bytes(img_name, images_dict, img_lookup_display)
            
            if img_bytes:
                st.image(img_bytes, use_column_width=True)
            else:
                st.error(f"Image '{img_name}' not found.")
            
            if "error" in item:
                st.error(f"Analysis Failed: {item['error']}")
            else:
                st.success(f"**Title:** {item.get('title_text', 'N/A')}")
                with st.expander("See Full Headline/Copy"):
                    st.write(item.get('headline_text', 'N/A'))
                st.info(f"**Background:** {item.get('bg_label', '-')}")
                st.info(f"**Contrast:** {item.get('contrast_label', '-')}")
                st.write(f"**Face:** {'Yes' if item.has_face else 'No'}")
                st.write(f"**Product Size:** {item.get('product_size_bucket', 'N/A')} ({item.get('product_area_pct', 0):.1f}%)")
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
    if not csv_file or not uploaded_images:
        st.error("Please upload both a CSV file and at least one Image before running.")
        st.stop()

    with st.spinner("Loading AI Models..."):
        face_cascade, ocr_reader = load_models()
        if not face_cascade: st.stop()

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

    images_dict = {f.name.strip(): f.getvalue() for f in uploaded_images}
    
    images_lookup = {}
    for name in images_dict.keys():
        images_lookup[name] = name 
        images_lookup[name.lower()] = name 
        name_no_ext = os.path.splitext(name)[0]
        images_lookup[name_no_ext] = name 
        images_lookup[name_no_ext.lower()] = name 

    all_features = []
    bar = st.progress(0, text="Analyzing...")
    total_rows = len(df)
    
    for i, row in df.iterrows():
        csv_name = row[image_name_col]
        
        target_key = None
        if csv_name in images_dict: target_key = csv_name
        elif csv_name.lower() in images_lookup: target_key = images_lookup[csv_name.lower()]
        elif csv_name in images_lookup: target_key = images_lookup[csv_name]
        elif csv_name.lower() in images_lookup: target_key = images_lookup[csv_name.lower()]
            
        if target_key:
            feats = analyze_image_features(images_dict[target_key], face_cascade, ocr_reader)
            all_features.append({**row.to_dict(), **feats, 'image_name': csv_name})
        
        bar.progress((i + 1) / total_rows)
    bar.empty()
    
    if not all_features:
        st.error("No matching images analyzed.")
        with st.expander("Debug: Mismatch Checker", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**CSV Filenames (First 5):**")
                st.write(df[image_name_col].head(5).tolist())
            with col_b:
                st.write("**Uploaded Filenames (First 5):**")
                st.write(list(images_dict.keys())[:5])
        st.stop()

    res_df = pd.DataFrame(all_features)
    
    if metric_col in res_df.columns:
        res_df[metric_col] = pd.to_numeric(res_df[metric_col], errors='coerce')
        res_df = res_df.sort_values(by=metric_col, ascending=False)
    
    valid_df = res_df[res_df.get('error').isna()] if 'error' in res_df.columns else res_df
    
    if not valid_df.empty and metric_col in valid_df.columns:
        mean_val = valid_df[metric_col].mean()
        col1, col2 = st.columns(2)
        col1.metric(label=f"Dataset Average {metric_col}", value=f"{mean_val:.4f}")
        col2.metric(label="Benchmark Used", value=f"{benchmark_val}")
        
        display_best_vs_worst(valid_df, metric_col, images_dict)

        display_aggregate_report(
            valid_df[valid_df[metric_col] > benchmark_val], 
            valid_df[valid_df[metric_col] <= benchmark_val], 
            metric_col,
            benchmark_val
        )
    else:
        st.warning("Analysis complete, but no valid metrics found to aggregate.")

    display_full_data(res_df, metric_col, image_name_col)
