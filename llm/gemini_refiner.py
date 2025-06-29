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

Your job is to classify ONLY the **relevant YouTube coding resources** below into meaningful learning categories, such as but not limited to:
- "For learning DSA"
- "For web development"
- "For DSA insights"
- "For DSA Motivation"
- "For DSA Strategy"
- "Bonus Content"
- "Not related to topic, but useful"

Every learning category should mandatorily have a resource name, channel name, description, video link. Otherwise do not make that category. Response should be extremely user friendly.

If "{query}" is unrelated to computer science then follow the SPECIAL RULE below **mandatorily**.

📌 You may also create your own relevant category names depending on the topic input, but:
- Do NOT use generic labels like "Uncategorized"
- Most categories should be **learning-oriented**
- One category can be "Bonus Content" or "For extra knowledge"

📌 Mandatory Guidelines:
- ✅ Prioritize actual technical learning resources over motivational or opinion-based videos.
- ✅ Do not leave any category empty.
- ✅ At least 1–2 resources must cover **subtopics** of the main query. (E.g., for "DSA", subtopics might include recursion, trees, linked lists, etc.)
- ✅ Ensure each **description is complete, grammatically correct, and ~2–3 lines long**. Do NOT cut mid-sentence.
- ❗If NONE of the provided videos are genuinely relevant to the query topic (even if topic sounds technical), return the special flag below.

---

🚨 SPECIAL RULE: If the input topic "{query}" is clearly **unrelated to computer science or coding** (e.g., "banana", "dating", "football", "balls"), then:
- ❗Invent **funny but serious-sounding educational categories** about "{query}" (e.g., "For Elite Ball Knowledge", "Banana Algorithms", etc.)
- ❗In every `"Video Link"` field, insert:
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ&start_radio=1"
- ❗Invent a **different, funny-but-convincing YouTube channel name** for each item (1–3 words, related to the query)
- ❗Invent a **funny, serious-sounding video title and description** about "{query}" for each resource
- ❗Still return output ONLY in the JSON format shown below — do not change the structure

---

🚫 If you believe **none of the resources are truly related to coding or computer science** (even though the "{query}" may sound technical), just return this exactly:
{{ "no_cs_data_found": true }}

---

📦 Return your final answer STRICTLY in this format (no markdown, no explanations, no commentary):

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

Use the title, description, and channel to infer the topic if unclear.

---

Here are the resources to classify:

{df_slice.to_json(orient="records", indent=2)}

⚠️ Output ONLY valid JSON. Do not include anything else. Do not try to be clever.
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
