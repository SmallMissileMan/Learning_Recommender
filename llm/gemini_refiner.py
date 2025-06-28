import google.generativeai as genai
import pandas as pd
import json
import os
import streamlit as st  # Assuming Streamlit is the front

# Use secrets.toml instead of dotenv for deployment
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

def refine_results(query, df):
    try:
        # Configure and instantiate Gemini model only when called
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        df_slice = df[["Resource Name", "Channel Name", "Description", "Video Link"]].head(20)

        prompt = f"""
Classify ONLY relevant YouTube coding resources below into meaningful learning categories like "For learning DSA", "For web development", "For DSA insights", "For DSA Motivation", "For DSA Strategy", etc. Make similar relevant categories by yourself. Omit resources that don't provide clear educational value.
Give more priority to videos i.e., show em at the top if they are related to learning some topic (instead of insights, motivation, etc...).
Each section must be a key, and the value must be a list of dictionaries like this:
[{{"Resource Name": "...", "Channel Name": "...", "Description": "...", "Video Link": "..."}}]

{df_slice.to_string(index=False)}
If the video seems unclear or vague, make your best guess about its topic based on title and channel type. Do not say "purpose is unclear".
Do not use triple backticks or markdown formatting. Output must be raw JSON only.
Make the description as informative as possible but like 3 lines long approximately.
And don't be oversmart and give any other response — it should be just this exactly, nothing extra, no BS, don't even give an intro or ending, just the answer in JSON format
        """

        # Call Gemini
        response = model.generate_content(prompt)

        # Raw text output
        cleaned = response.text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        return json.loads(cleaned)

    except Exception as e:
        print("❌ Gemini Refiner Error:", str(e))
        return df  # Fallback to ungrouped results
