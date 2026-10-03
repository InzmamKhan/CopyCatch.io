import streamlit as st
from core.models import DuplicateGroup
from core.image_loader import ImageLoader

def render_group_card(group: DuplicateGroup):
    """Renders a compact grid card with half-sized image previews for a duplicate group."""
    st.markdown(f"#### Group {group.group_id} ({len(group.images)} Duplicate Images)")
    
    # Increase column count per row to shrink individual image column width
    cols = st.columns(min(len(group.images) * 2, 8))
    
    for idx, img_item in enumerate(group.images):
        col = cols[idx * 2]  # Double spacing to constrain width
        with col:
            thumb = ImageLoader.create_thumbnail(img_item.path)
            # Capped width at 150px for half-size previews
            st.image(thumb, width=150)
            st.caption(f"**{img_item.filename}**")
            st.text(f"Path: {img_item.path.parent.name}/")
    st.divider()