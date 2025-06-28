import google.generativeai as genai
import pandas as pd
import json
import os
import streamlit as st

# ✅ Get API key safely from Streamlit Secrets or Env
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

# ✅ Gemini-based result refinement
def refine_results(query, df):
    try:
        st.write("🔧 Starting Gemini refinement...")  # ✅ Start log
        if not api_key:
            st.warning("⚠️ Gemini API key not found. Showing default recommendations.")
            return df  # fallback

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        # Use top 20 results for refinement
        st.write("📊 Preparing top 20 results...")
        df_slice = df[["Resource Name", "Channel Name", "Description", "Video Link"]].head(20)
        st.write("📜 Building prompt...")
        prompt = f"""
Classify ONLY relevant YouTube coding resources below into meaningful learning categories like:
"For learning DSA", "For web development", "For DSA insights", "For DSA Motivation", "For DSA Strategy", etc.
Come up with your own relevant categories too.

Prioritize actual technical learning resources over motivational ones. Do not leave any category empty.

Format:
{{
    "Category 1": [{{"Resource Name": "...", "Channel Name": "...", "Description": "...", "Video Link": "..."}}],
    "Category 2": [{{...}}]
}}

Use the title and channel to guess the topic if unclear. Keep the description informative (~3 lines). Do NOT include markdown or backticks, and don’t say anything else outside the JSON.
        
{df_slice.to_string(index=False)}
        """
        st.write("🚀 Sending to Gemini...")
        # 🔁 Generate content
        response = model.generate_content(prompt)
        cleaned = response.text.strip()
        st.write("✅ Gemini response received.")

        # ✅ Sanitize output
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned)
        st.write("📦 Parsed JSON categories:", list(parsed.keys()))
        return parsed

    except Exception as e:
        print("❌ Gemini Refiner Error:", str(e))
        st.error("Gemini failed to refine results. Showing default output.")
        return df  # fallback
