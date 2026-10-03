import streamlit as st
import time
from ui.components.folder_picker import open_folder_dialog
from ui.components.stats_bar import render_stats_bar
from ui.components.group_card import render_group_card
from pipeline.detector import DuplicateDetectorPipeline
from config.settings import settings

def render():
    st.set_page_config(page_title="CopyCatch.io", layout="wide")
    st.title("CopyCatch.io — Sub-Millisecond Duplicate Detector")

    if "selected_folder" not in st.session_state:
        st.session_state.selected_folder = ""
    if "results" not in st.session_state:
        st.session_state.results = None

    # Top Folder Selection Panel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("Target Folder", value=st.session_state.selected_folder, disabled=True)
    with col2:
        st.write(" ")
        if st.button("📁 Select Folder"):
            folder = open_folder_dialog()
            if folder:
                st.session_state.selected_folder = folder
                st.session_state.results = None

    # Advanced Scan Parameters Setup
    with st.expander("⚙️ Advanced Scan Settings & Filters", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            direct_thresh = st.slider(
                "Direct Match Hamming Threshold", 
                min_value=0, max_value=10, 
                value=settings.DIRECT_DUPLICATE_THRESHOLD,
                help="Maximum Hamming distance to consider images an exact duplicate instantly."
            )
        with col_b:
            ambiguous_thresh = st.slider(
                "Ambiguous Hamming Search Threshold", 
                min_value=6, max_value=25, 
                value=settings.AMBIGUOUS_THRESHOLD,
                help="Upper Hamming distance limit to trigger ONNX neural verification."
            )
        with col_c:
            cosine_thresh = st.slider(
                "ONNX Cosine Similarity Cutoff", 
                min_value=0.50, max_value=0.99, 
                value=float(settings.COSINE_SIMILARITY_THRESHOLD), 
                step=0.01,
                help="Minimum visual similarity required from ONNX embeddings to group images."
            )

        selected_exts = st.multiselect(
            "Target Extensions",
            options=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
            default=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
            help="Filter which file types CopyCatch will process."
        )

    if st.session_state.selected_folder:
        if st.button("🚀 Scan for Duplicates", type="primary"):
            # Update settings dynamically for current run
            settings.DIRECT_DUPLICATE_THRESHOLD = direct_thresh
            settings.AMBIGUOUS_THRESHOLD = ambiguous_thresh
            settings.COSINE_SIMILARITY_THRESHOLD = cosine_thresh
            settings.SUPPORTED_EXTENSIONS = tuple(selected_exts)

            with st.spinner("Analyzing image hashes & searching for duplicates..."):
                start_time = time.time()
                pipeline = DuplicateDetectorPipeline()
                dup_groups = pipeline.run(st.session_state.selected_folder)
                elapsed = time.time() - start_time
                
                st.session_state.results = (dup_groups, elapsed)

    # Display Duplicate Groups
    if st.session_state.results:
        dup_groups, elapsed = st.session_state.results
        total_dup_images = sum(len(g.images) for g in dup_groups)
        
        render_stats_bar(
            total_groups=len(dup_groups),
            total_duplicate_images=total_dup_images,
            execution_time=elapsed
        )
        
        st.markdown("---")
        st.subheader("Duplicate Clusters")
        
        if not dup_groups:
            st.info("No duplicates detected in this directory with the current settings.")
        else:
            for group in dup_groups:
                render_group_card(group)