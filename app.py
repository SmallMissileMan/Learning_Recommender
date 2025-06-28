import streamlit as st
import pandas as pd
from ml.model import load_dataset, encode_with_sentence_transformer, recommend_resources
from llm.gemini_refiner import refine_results

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Smart Video Recommender", layout="wide")

# --- Custom CSS for Dark Mode & Visibility ---
st.markdown("""
    <style>
        body, .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }

        .stTextInput>div>div>input,
        .stSlider>div>div>div>input,
        .stNumberInput>div>div>input {
            color: #FFFFFF !important;
            background-color: #333333 !important;
        }

        .css-1aumxhk {
            color: #FFFFFF !important;
        }

        label,
        .stTextInput label,
        .stSlider label,
        .stNumberInput label,
        .css-1cpxqw2, .css-1cpxqw2 p, .css-1cpxqw2 label {
            color: #FFFFFF !important;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #FFFFFF;
        }

        a {
            color: #1E90FF !important;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Data and Encode ---
st.title("🔍 Smart Coding Video Recommender")
st.markdown("**Search smarter, not harder.** This app uses semantic search (not just keywords!) to fetch the most relevant coding resources from YouTube – organized, ranked, and LLM-enhanced.")

with st.spinner("Loading video database and model..."):
    df = load_dataset()
    embeddings = encode_with_sentence_transformer(df)

# --- Input UI ---
query = st.text_input("🔍 Ask for a topic or concept you're learning:", "")
num_results = st.slider("🧠 Number of results:", min_value=1, max_value=10, value=5)

# --- Search and Display ---
if query:
    top_df = recommend_resources(query, df, embeddings, top_n=num_results)
    grouped = refine_results(query, top_df)

    if isinstance(grouped, dict) and grouped == {"no_cs_data_found": True}:
        st.warning("⚠️ No relevant resources found for this topic in our dataset.")
    elif isinstance(grouped, dict):
        for section_title, videos in grouped.items():
            st.subheader(f"📂 {section_title}")
            for row in videos:
                short_desc = row.get('Description', '')[:200] + ('...' if len(row.get('Description', '')) > 200 else '')
                video_link = row.get('Video Link', '#')
                title = row.get('Resource Name', 'Untitled')
                st.markdown(f"🎥 [**{title}**]({video_link})")
                st.markdown(f"📡 {row.get('Channel Name', 'Unknown Channel')}")
                st.markdown(f"📝 {short_desc}")
                st.markdown("---")
    else:
        st.subheader("📂 Top Results")
        for _, row in top_df.iterrows():
            short_desc = row.get('Description', '')[:200] + ('...' if len(row.get('Description', '')) > 200 else '')
            video_link = row.get('Video Link', '#')
            title = row.get('Resource Name', 'Untitled')
            st.markdown(f"🎥 [**{title}**]({video_link})")
            st.markdown(f"📡 {row.get('Channel Name', 'Unknown Channel')}")
            st.markdown(f"📝 {short_desc}")
            st.markdown("---")

# --- About Section ---
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("This app helps learners discover the most relevant YouTube coding resources using semantic search (via sentence transformers) and LLM-based refinement for contextual ranking.")
