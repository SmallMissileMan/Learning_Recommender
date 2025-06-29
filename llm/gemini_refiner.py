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
        prompt = f"""
You are a helpful and smart educational video classifier assistant.

A user is looking for resources related to: "{query}"

Your job is to classify ONLY the **relevant YouTube coding resources** below into clear, meaningful learning categories. Examples include (but are not limited to):
- "For learning DSA"
- "For web development"
- "For DSA insights"
- "For DSA motivation"
- "Bonus Content"
- "Not related to topic, but useful"

📌 VERY IMPORTANT:

❗You MUST follow these rules:
- ❌ Do NOT create a category if even a single resource inside it is missing any of these:
  - `Resource Name`
  - `Channel Name`
  - `Description` (must be 2–3 lines, grammatically correct, and **must NOT end with '...'**)
  - `Video Link` (must be a valid YouTube link)
- ✅ Only include categories that contain **only valid items** as above.
- ✅ Do NOT truncate any field.
- ✅ Do NOT leave any category empty.
- ✅ Drop the entire category if even one item inside it is invalid or incomplete.

🎯 Your goal is to give a very **user-friendly**, **organized**, and **refined** classification that feels human-curated. Do NOT be lazy.

📌 You may invent your own smart category names depending on the input, but:
- Do NOT use vague labels like "Uncategorized"
- Most categories should be **learning-focused**
- One category can be "Bonus Content" or "Extra Insight"

📌 Technical Guidelines:
- ✅ Prioritize learning/educational content over motivational or commentary videos.
- ✅ Include subtopics if possible. E.g., if topic is "DSA", include resources on recursion, trees, sorting, etc.
- ✅ Never include clickbait or joke videos unless query is non-CS.

---

🚨 SPECIAL RULE: If the input topic "{query}" is clearly **not related to computer science or coding** (e.g., "banana", "football", "balls"):
- ❗Create fake but serious-sounding educational categories (e.g., "Banana Algorithms", "Ball Theory")
- ❗Make up funny but believable channel names (e.g., "Fruit Tech", "Playball Weekly")
- ❗Use this exact YouTube link for every result:
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1"
- ❗Fabricate a detailed, quirky-but-plausible description for each item (~2–3 lines, grammatically correct)
- ❗Still return ONLY valid JSON. No markdown, no explanation.

---

🚫 If you determine that NONE of the resources are truly relevant to coding or computer science, even if the topic seems technical, return this JSON exactly:
{{ "no_cs_data_found": true }}

---

📦 Output your final response in this **strict JSON format** (NO markdown, NO text, NO prefix, NO suffix):

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

Use title, description, and channel to infer the video’s topic. Be strict.

---

Here are the resources to classify:

{df_slice.to_json(orient="records", indent=2)}

⚠️ Output **ONLY valid JSON**. Nothing else. Do not try to be clever.
"""

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
