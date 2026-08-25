import streamlit as st
import os
import importlib
import sys

# Force reload main to prevent Streamlit from caching old mock functions/strings
import main
importlib.reload(main)

# Also force reload sub-modules
import get_response
importlib.reload(get_response)

from main import run_pipeline

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

SAMPLES = [
    {"label": "Healthcare: Fever and basic care advice (sample_01.wav)", "filename": "sample_01.wav"},
    {"label": "Healthcare: Symptoms of common cold (sample_02.wav)", "filename": "sample_02.wav"},
    {"label": "Healthcare: How to book a doctor's appointment (sample_03.wav)", "filename": "sample_03.wav"},
    {"label": "Healthcare: Vaccination schedule for children (sample_04.wav)", "filename": "sample_04.wav"},
    {"label": "Healthcare: What to do in case of snake bite (sample_05.wav)", "filename": "sample_05.wav"},
    {"label": "Healthcare: Maternal health checkup reminder (sample_06.wav)", "filename": "sample_06.wav"},
    {"label": "Healthcare: Diabetes diet advice (sample_07.wav)", "filename": "sample_07.wav"},
    {"label": "Governance: How to apply for a ration card (sample_08.wav)", "filename": "sample_08.wav"},
    {"label": "Governance: Checking land record status (sample_09.wav)", "filename": "sample_09.wav"},
    {"label": "Governance: Filing a village-level grievance (sample_10.wav)", "filename": "sample_10.wav"},
    {"label": "Governance: Applying for a birth certificate (sample_11.wav)", "filename": "sample_11.wav"},
    {"label": "Governance: Information on government scholarship scheme (sample_12.wav)", "filename": "sample_12.wav"},
    {"label": "Governance: How to pay electricity bill online (sample_13.wav)", "filename": "sample_13.wav"},
    {"label": "Governance: Reporting a road or infrastructure issue (sample_14.wav)", "filename": "sample_14.wav"},
    {"label": "Governance: Checking voter ID registration status (sample_15.wav)", "filename": "sample_15.wav"},
]


st.set_page_config(
    page_title="Northeast Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)

# Custom premium warm paper design styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;0,700;1,400&display=swap');

/* Force background color on main Streamlit body container and header */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #F5EFE6 !important;
    color: #1A1A1A !important;
}

/* Hide Streamlit default header to reclaim space */
[data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
}

/* Reduce top padding of main block container */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 46rem !important;
}

/* Force body typography and color */
.stApp, p, span, li, button, input, label, textarea {
    font-family: 'Inter', sans-serif !important;
    color: #1A1A1A !important;
}

/* Heading Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Lora', serif !important;
    color: #2B4C3F !important;
    font-weight: 700 !important;
}

/* Custom Header with warm borders */
.header-container {
    text-align: center;
    padding: 2.2rem 1.5rem;
    background-color: rgba(43, 76, 63, 0.04);
    border-radius: 20px;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(43, 76, 63, 0.12);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2B4C3F !important;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    font-family: 'Lora', serif !important;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #8B7355 !important; /* Bronze muted text */
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.5;
}

/* Language Pill Strip */
.language-strip {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 2.5rem;
    margin-top: -0.5rem;
}

.lang-pill {
    background-color: rgba(43, 76, 63, 0.06);
    color: #2B4C3F !important; /* Pine Green */
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    font-family: 'Inter', sans-serif !important;
    border: 1px solid rgba(43, 76, 63, 0.15);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
}

/* Paper-style card styling */
.paper-card {
    background-color: #FFFFFF; /* Pure white card for paper contrast */
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid rgba(139, 115, 85, 0.2); /* Muted bronze border */
    margin-top: 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 12px rgba(139, 115, 85, 0.04);
    transition: all 0.3s ease;
}

.paper-card:hover {
    transform: translateY(-2px);
    border-color: #B5502D; /* Terracotta border on hover */
    box-shadow: 0 8px 20px rgba(181, 80, 45, 0.08);
}

.card-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #B5502D; /* Terracotta/copper secondary accent */
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Inter', sans-serif !important;
}

.card-content {
    font-size: 1.1rem;
    line-height: 1.6;
    font-weight: 500;
    color: #1A1A1A !important;
    font-family: 'Inter', sans-serif !important;
}

/* Custom Selectbox styling for light/dark mode compatibility */
div[data-testid="stSelectbox"] > div {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border-radius: 8px !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1px solid rgba(139, 115, 85, 0.3) !important;
    border-radius: 8px !important;
}

/* Force selected text to be black and visible (high specificity overrides) */
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] div,
div[data-testid="stSelectbox"] input,
div[data-testid="stSelectbox"] p,
div[data-testid="stSelectbox"] li,
div[data-testid="stSelectbox"] label {
    color: #1A1A1A !important;
    background-color: transparent !important;
}

/* Options dropdown container and items */
div[data-baseweb="popover"] ul, [data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

div[data-baseweb="popover"] li, [data-baseweb="menu"] li {
    color: #1A1A1A !important;
    background-color: #FFFFFF !important;
}

/* Options dropdown container and items text color (high specificity overrides) */
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] div,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] p,
[data-baseweb="menu"] span,
[data-baseweb="menu"] div,
[data-baseweb="menu"] li,
[data-baseweb="menu"] p {
    color: #1A1A1A !important;
}

div[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover {
    background-color: rgba(43, 76, 63, 0.08) !important;
}

/* Custom stButton (Process Audio Pipeline button) styling */
div[data-testid="stButton"] button {
    background-color: #2B4C3F !important; /* Premium Pine Green */
    color: #FFFFFF !important;            /* White text */
    border: 1px solid #2B4C3F !important;
    border-radius: 100px !important;       /* Pill shape */
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(43, 76, 63, 0.15) !important;
}

div[data-testid="stButton"] button:hover {
    background-color: #B5502D !important; /* Terracotta on hover */
    border-color: #B5502D !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 15px rgba(181, 80, 45, 0.2) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stButton"] button:active {
    transform: translateY(1px) !important;
}

/* Ensure the button text color is white and overrides global font color */
div[data-testid="stButton"] button, 
div[data-testid="stButton"] button p, 
div[data-testid="stButton"] button span {
    color: #FFFFFF !important;
}

/* Visual match style for st.audio_input (dark-gray capsule container) */
div[data-testid="stAudioInput"] {
    background-color: #303030 !important;
    border-radius: 40px !important;
    padding: 6px 16px !important;
    height: 54px !important;
    border: none !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
}

/* Hide internal default label inside st.audio_input */
div[data-testid="stAudioInput"] label {
    display: none !important;
}

/* Adjust inner padding of st.audio_input wrapper */
div[data-testid="stAudioInput"] > div {
    padding: 0 !important;
    margin: 0 !important;
    background-color: transparent !important;
}

/* Style action buttons inside st.audio_input (record/stop/play) to look like native controls */
div[data-testid="stAudioInput"] button {
    background-color: transparent !important;
    color: #CCCCCC !important;
    border: none !important;
    box-shadow: none !important;
    padding: 6px !important;
    margin: 0 4px !important;
    width: auto !important;
    height: auto !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stAudioInput"] button:hover {
    color: #FFFFFF !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* Standardize svg icon size and color */
div[data-testid="stAudioInput"] button svg {
    fill: currentColor !important;
    stroke: currentColor !important;
    width: 20px !important;
    height: 20px !important;
}

/* Strip any default borders, outlines or box shadows from ALL child elements inside st.audio_input */
div[data-testid="stAudioInput"] * {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Hide any SVG rect elements that might draw square borders around icons */
div[data-testid="stAudioInput"] svg rect {
    display: none !important;
}

/* Style timer text and other text fields inside st.audio_input to match st.audio style */
div[data-testid="stAudioInput"] span, 
div[data-testid="stAudioInput"] p, 
div[data-testid="stAudioInput"] div {
    color: #CCCCCC !important;
    font-size: 0.85rem !important;
    font-family: monospace !important;
}

/* Style the audio input waveform to render in a light-gray progress line look */
div[data-testid="stAudioInput"] canvas {
    filter: brightness(0) invert(0.8) !important; /* Converts green/red waves to light gray */
    height: 24px !important;
    opacity: 0.65 !important;
}

/* Style st.audio output player container to match */
div[data-testid="stAudio"] {
    background-color: #303030 !important;
    border-radius: 40px !important;
    padding: 6px 16px !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stAudio"] audio {
    filter: invert(1) hue-rotate(180deg) !important; /* Force native browser player into dark gray matching style */
    width: 100%;
    height: 40px !important;
}

</style>
""", unsafe_allow_html=True)

# Main UI title and introduction
st.markdown("""
<div class="header-container">
    <div class="header-title">Northeast Voice Assistant</div>
    <div class="header-subtitle">AI-powered multilingual voice assistant for healthcare and governance queries in Northeast Indian languages</div>
</div>
""", unsafe_allow_html=True)

# Static Language Pill Strip
st.markdown("""
<div class="language-strip">
    <span class="lang-pill">Assamese</span>
</div>
""", unsafe_allow_html=True)

st.subheader("🎙️ Ask your Question")

st.markdown("<p style='color: #8B7355; font-family: \"Inter\", sans-serif; font-weight: 500; margin-bottom: 0.6rem; font-size: 0.95rem;'>Choose a pre-recorded Assamese sample or upload your own WAV file:</p>", unsafe_allow_html=True)

# Selection list
options = ["-- Choose an Option --", "Upload your own WAV file..."]
for i, sample in enumerate(SAMPLES, 1):
    options.append(f"Sample {i:02d}: {sample['label']}")

selected_option = st.selectbox(
    "Choose input source:",
    options,
    label_visibility="collapsed"
)

audio_path_to_run = None
uploaded_bytes = None

if selected_option == "Upload your own WAV file...":
    uploaded_file = st.file_uploader("Upload a WAV audio file:", type=["wav"])
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.read()
elif selected_option != "-- Choose an Option --":
    idx_str = selected_option.split("Sample ")[1].split(":")[0]
    idx = int(idx_str) - 1
    sample = SAMPLES[idx]
    audio_path_to_run = os.path.join(PROJECT_ROOT, "tts", "outputs", sample["filename"])

if audio_path_to_run or uploaded_bytes:
    st.markdown("🔊 **Listen to the Input Query:**")
    if audio_path_to_run:
        with open(audio_path_to_run, "rb") as f:
            st.audio(f.read(), format="audio/wav")
    else:
        st.audio(uploaded_bytes, format="audio/wav")
    
    run_btn = st.button("🚀 Process Audio Pipeline", use_container_width=True)
    
    if run_btn:
        import tempfile
        import uuid
        
        temp_dir = tempfile.gettempdir()
        unique_id = uuid.uuid4().hex
        input_path = os.path.join(temp_dir, f"input_{unique_id}.wav")
        output_path = os.path.join(temp_dir, f"output_{unique_id}.wav")
        
        try:
            if audio_path_to_run:
                with open(audio_path_to_run, "rb") as fsrc:
                    input_bytes = fsrc.read()
            else:
                input_bytes = uploaded_bytes
                
            with open(input_path, "wb") as fdst:
                fdst.write(input_bytes)
                
            with st.spinner("Processing Voice Pipeline (ASR → NLU → TTS)..."):
                result = run_pipeline(input_path, output_path)
            
            transcribed_text = result.get("transcribed_text")
            reply_text = result.get("reply_text")
            output_audio_path = result.get("audio_output_path")
            
            if not transcribed_text:
                st.error("Couldn't transcribe audio. Please try another sample or file.")
            elif not reply_text:
                st.error("Assistant was unable to generate a response. Please try again.")
            else:
                st.markdown(f"""
                <div class="paper-card">
                    <div class="card-label">📝 Transcribed Question</div>
                    <div class="card-content">"{transcribed_text}"</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="paper-card">
                    <div class="card-label">🤖 Assistant Response</div>
                    <div class="card-content">{reply_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if output_audio_path and os.path.exists(output_audio_path):
                    with open(output_audio_path, "rb") as fout:
                        st.markdown("🔊 **Listen to the Response:**")
                        st.audio(fout.read(), format="audio/wav")
                else:
                    st.warning("TTS audio output could not be generated.")
                    
        except Exception as e:
            print(f"Error running pipeline in Streamlit: {e}", file=sys.stderr)
            st.error(f"Error running pipeline: {e}")
            
        finally:
            if os.path.exists(input_path):
                try: os.remove(input_path)
                except: pass
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except: pass

