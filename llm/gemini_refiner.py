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
You are a helpful learning assistant. Classify ONLY relevant YouTube coding resources below into meaningful learning categories such as:
"For learning DSA", "For web development", "For DSA insights", "For DSA Motivation", "For DSA Strategy", etc.
You may create your own relevant categories as needed — all categories must be learning-oriented.

🔹 Prioritize actual technical learning resources over motivational or opinion content.
🔹 Do not leave any category empty.
🔹 At least one or two resources must cover **subtopics related to the user's query** (e.g., if query is "DSA", subtopics could include recursion, trees, greedy, linked lists, etc.).
🔹 For each resource, ensure the **description is grammatically correct and complete**, ideally around 2–3 lines long. Do NOT cut descriptions mid-sentence.

Return the final result strictly in the following JSON format (with no markdown, no commentary, no backticks):
{{
  "Category 1": [
    {{
      "Resource Name": "...",
      "Channel Name": "...",
      "Description": "...",
      "Video Link": "..."
    }}
  ],
  "Category 2": [
    {{
      ...
    }}
  ]
}}

Use the video title, description, and channel name to infer topics if unclear.

Here are the resources to classify:
        
{df_slice.to_string(index=False)}
Do not write anything else, just give me the answer in json format as mentioned above. Do not act oversmart.
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
