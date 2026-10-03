import streamlit as st
import time
from ui.components.folder_picker import open_folder_dialog
from ui.components.stats_bar import render_stats_bar
from ui.components.group_card import render_group_card
from pipeline.detector import DuplicateDetectorPipeline

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

    if st.session_state.selected_folder:
        if st.button("🚀 Scan for Duplicates", type="primary"):
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
            st.info("No duplicates detected in this directory.")
        else:
            for group in dup_groups:
                render_group_card(group)