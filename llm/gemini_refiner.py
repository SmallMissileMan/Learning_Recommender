import google.generativeai as genai
import pandas as pd
import json
import os
import streamlit as st

# ✅ Get API key safely from Streamlit Secrets or Env
api_key = st.secrets.get("gemini", {}).get("api_key") or os.getenv("GEMINI_API_KEY")

# ✅ Gemini-based result refinement
def refine_results(query, df):
    try:
        if not api_key:
            st.warning("⚠️ Gemini API key not found. Showing default recommendations.")
            return df  # fallback

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        # 🔍 Step 1: Check if the query is related to CS
        check_prompt = f"""
Is the topic "{query}" clearly related to computer science, programming, or software engineering?

Reply ONLY with "Yes" or "No".
"""
        check_response = model.generate_content(check_prompt)
        is_cs_related = check_response.text.strip().lower().startswith("yes")

        # 🐸 Step 2: If unrelated to CS, fabricate dummy dataset to force SPECIAL RULE
        if not is_cs_related:
            df = pd.DataFrame([{
                "Resource Name": "placeholder",
                "Channel Name": "placeholder",
                "Description": "placeholder",
                "Video Link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            } for _ in range(5)])

        # Use top 20 results for refinement
        df_slice = df[["Resource Name", "Channel Name", "Description", "Video Link"]].head(20)
        prompt = f"""<your long prompt here, unchanged>"""  # for brevity

        # 🔁 Generate content
        response = model.generate_content(prompt)
        cleaned = response.text.strip()

        # ✅ Sanitize output
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        parsed = json.loads(cleaned)

        # ✅ Check if Gemini returned flag or empty categories
        if isinstance(parsed, dict):
            if "no_cs_data_found" in parsed:
                return {"no_cs_data_found": True}
            if all(isinstance(v, list) and len(v) == 0 for v in parsed.values()):
                return {"no_cs_data_found": True}

        return parsed

    except Exception as e:
        print("❌ Gemini Refiner Error:", str(e))
        st.error("Gemini failed to refine results. Showing default output.")
        return df  # fallback
