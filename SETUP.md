# Local Setup

1. Clone the repo
2. Create virtual environment:
   python -m venv .venv
   .venv\Scripts\activate   (Windows)  |  source .venv/bin/activate  (Mac/Linux)
3. Install dependencies:
   pip install -r requirements.txt
4. Copy nlu-mt/.env.example to nlu-mt/.env and add your NVIDIA_API_KEY
5. Run:
   streamlit run app/streamlit_app.py

First run downloads ASR/TTS models from Hugging Face (~2-3GB) — requires internet, takes a few minutes.
