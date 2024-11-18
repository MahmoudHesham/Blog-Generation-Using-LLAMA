import streamlit as st
import model as ml

## Streamlit UI Setup
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="💃",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("Generate Blogs 💃")

blog_topic = st.text_input("Blog Topic")

number_of_words_col, blog_target_audience_col = st.columns([5, 5])

with number_of_words_col:
    number_of_words = st.text_input("Number of Words")

with blog_target_audience_col:
    blog_target_audience = st.selectbox(
        "Audience",
        ("Fashion Enthusiasts", "Artists", "Researchers", "Normal People"),
        index=0,
    )

use_gpu = st.checkbox("Use GPU", value=False, label_visibility="visible")
generate_button = st.button("Generate")

if generate_button:
    st.write(
        ml.get_model_response(
            blog_topic, number_of_words, blog_target_audience, use_gpu
        )
    )
