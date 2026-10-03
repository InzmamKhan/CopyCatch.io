import streamlit as st

def render_stats_bar(total_groups: int, total_duplicate_images: int, execution_time: float):
    """Renders a summary metrics bar for scan execution time and duplicate counts."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Duplicate Groups Found", value=total_groups)
    with col2:
        st.metric(label="Total Duplicate Images", value=total_duplicate_images)
    with col3:
        st.metric(label="Scan Time", value=f"{execution_time:.3f} s")