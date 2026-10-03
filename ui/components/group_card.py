import streamlit as st
from core.models import DuplicateGroup
from core.image_loader import ImageLoader

def render_group_card(group: DuplicateGroup):
    """Renders a grid card containing image previews and filenames for a duplicate group."""
    st.markdown(f"#### Group {group.group_id} ({len(group.images)} Duplicate Images)")
    
    cols = st.columns(min(len(group.images), 4))
    
    for idx, img_item in enumerate(group.images):
        col = cols[idx % 4]
        with col:
            thumb = ImageLoader.create_thumbnail(img_item.path)
            st.image(thumb, use_container_width=True)
            st.caption(f"**{img_item.filename}**")
            st.text(f"Path: {img_item.path.parent.name}/")
    st.divider()