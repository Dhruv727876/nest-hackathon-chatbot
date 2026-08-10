import streamlit as st
import os
import importlib
import sys

# Force reload main to prevent Streamlit from caching old mock functions/strings
import main
importlib.reload(main)
from main import speech_to_text, get_response, text_to_speech

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
    <span class="lang-pill">Bodo</span>
    <span class="lang-pill">Khasi</span>
    <span class="lang-pill">Manipuri</span>
    <span class="lang-pill">Nagamese</span>
</div>
""", unsafe_allow_html=True)

st.subheader("🎙️ Ask your Question")

# Render label/instruction outside and above the audio input bar
st.markdown("<p style='color: #8B7355; font-family: \"Inter\", sans-serif; font-weight: 500; margin-bottom: 0.6rem; font-size: 0.95rem;'>Speak your query (healthcare or governance) in your local language:</p>", unsafe_allow_html=True)

# Audio input widget for microphone capture with collapsed/hidden label
audio_file = st.audio_input(
    "Speak your query (healthcare or governance) in your local language:",
    label_visibility="collapsed"
)

if audio_file is not None:
    st.info("Processing voice query...")
    
    # Static temporary file paths to avoid locking issues with OS temp directories
    input_path = "input_temp.wav"
    output_path = "output_temp.wav"
    
    try:
        # Save the uploaded/recorded audio file to input_temp.wav
        input_saved = False
        try:
            with open(input_path, "wb") as f:
                f.write(audio_file.read())
            input_saved = True
        except Exception as e:
            print(f"Error saving input audio: {e}", file=sys.stderr)
            st.error("Failed to save audio input. Please try again.")
            
        if input_saved:
            # 1. Speech-to-Text Transcription
            transcribed_text = None
            try:
                transcribed_text = speech_to_text(input_path)
            except Exception as e:
                print(f"Error in speech_to_text: {e}", file=sys.stderr)
                
            # If speech_to_text() returns empty/None → show "Couldn't hear you, please try again"
            if not transcribed_text:
                st.error("Couldn't hear you, please try again")
            else:
                # 2. Get Response (NLU chatbot processing)
                reply_text = None
                try:
                    reply_text = get_response(transcribed_text)
                except Exception as e:
                    print(f"Error in get_response: {e}", file=sys.stderr)
                    
                # If get_response() fails or times out → show "Assistant is unavailable, please retry"
                if not reply_text:
                    st.error("Assistant is unavailable, please retry")
                else:
                    # Display the results using beautifully styled paper cards
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
                    
                    # 3. Text-to-Speech Synthesis
                    tts_success = False
                    try:
                        text_to_speech(reply_text, output_path)
                        tts_success = True
                    except Exception as e:
                        print(f"Error in text_to_speech: {e}", file=sys.stderr)
                        
                    # If text_to_speech() fails → still show text reply even if audio fails
                    if tts_success:
                        try:
                            # Read the generated output audio file
                            with open(output_path, "rb") as f:
                                audio_bytes = f.read()
                                
                            # Play the generated audio file
                            st.markdown("🔊 **Listen to the Response:**")
                            st.audio(audio_bytes, format="audio/wav")
                        except Exception as e:
                            print(f"Error reading/playing output audio: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        st.error(f"An error occurred during audio processing: {e}")
    finally:
        # Clean up temporary audio files safely
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
