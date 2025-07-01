💡 Smart Coding Video Recommender  
A next-gen coding content engine that combines **semantic understanding** and **LLM reasoning** to recommend the highest-quality YouTube learning videos. Uses sentence-transformers for similarity and Gemini to refine, clean, and classify your results. Streamlined. Smart. Stupid fast.

🔍 Features  
🧠 Sentence-Transformer Based Semantic Search

Embeds video descriptions using `all-MiniLM-L6-v2`  
Matches your query meaningfully using cosine similarity  
🤖 Gemini LLM Refinement

Filters out low-quality or irrelevant results  
Categorizes videos like:-  
• For learning DSA  
• For brushing up fundamentals  
• For advanced learners  
🎛️ Dynamic Output Control

User can choose between 1 to 10 results 
Fallback message if not enough high-quality matches found  
🖼️ Streamlit Interface

Fast, scrollable UI with clickable YouTube links  
Each card shows title, tags, and brief summary  

🛠️ Tech Stack:-  
Python  
Streamlit  
pandas  
scikit-learn  
numpy  
sentence-transformers  
google-generativeai (Gemini API)  
dotenv  

📁 Project Structure:-  
smart-coding-recommender/
├── dataset/
│   └── learning_resources.csv          # Cleaned YouTube metadata
├── llm/
│   └── gemini_refiner.py               # Gemini LLM filtering & tagging
├── ml/
│   └── model.py                        # Embedding + cosine similarity logic
├── scraping/
│   └── scraper.py                      # (Optional) YouTube data scraper
├── app.py                              # Main Streamlit frontend
├── precompute_embeddings.py            # One-time embedding generator
├── requirements.txt                    # Python dependencies
├── packages.txt                        # For Streamlit Cloud compatibility
└── .gitignore                          # Hides .env, cache, etc.

🧠 How It Works:-
User inputs a query → it's embedded using sentence-transformers
→ Top 20 semantically similar video descriptions are retrieved
→ Gemini LLM filters & classifies those results
→ Streamlit displays only the highest quality ones

✅ Example Prompts:-  
Learn recursion and backtracking
Best YouTube videos for Java roadmap
Understand Computer Networks visually
Frontend roadmap 2024
Quick DBMS revision videos

Prerequisites:-   
Python 3.8+
Gemini API Key
Streamlit

Installation:-  
git clone https://github.com/yourusername/smart-coding-recommender.git  
cd smart-coding-recommender  
pip install -r requirements.txt  
streamlit run app.py

Optional:-  
python precompute_embeddings.py     # Pre-generate video embeddings

🔐 Environment Setup:-  
Create a .env file and add:
GEMINI_API_KEY=your-secret-api-key

🧑‍💻 Author:-
Aaryan Madhu
MIT Manipal | CCE
Smarter tools for self-learners, no fluff just signal 🔥
