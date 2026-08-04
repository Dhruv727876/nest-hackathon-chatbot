import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file if available
load_dotenv()

try:
    from nlu_mt.assamese_examples import ASSAMESE_EXAMPLES
except ImportError:
    from assamese_examples import ASSAMESE_EXAMPLES

# Configurable model name for NVIDIA NIM API
# Alternative options to try:
# - "meta/llama-3.1-8b-instruct" (default)
# - "google/gemma-2-9b-it" (fallback)
# - "sarvamai/sarvam-m" (deprecated/dead)
# Can be overridden via NIM_MODEL_NAME environment variable
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")

def get_response(text: str) -> str:
    """
    Calls NVIDIA NIM API (Gemma model) to retrieve a response for healthcare/governance queries.
    
    System prompt: "You are a healthcare and governance assistant for Northeast India, specifically Assam. 
    ONLY answer questions about healthcare, medical symptoms, government schemes, or governance procedures. 
    If the question is about anything else (tourism, entertainment, general knowledge, etc.), 
    respond ONLY with: 'I can only help with healthcare and governance questions. Please ask about medical concerns or government schemes.' in Assamese.
    Do not answer off-topic questions even partially.
    Users may mix languages (Hindi, English, Assamese) in their question. Understand the intent regardless of language mixing, and still respond in Assamese.
    Respond strictly in Assamese (অসমীয়া), NOT Bengali. Assamese has distinct 
    features from Bengali — e.g., Assamese uses 'ৰ' (ra) instead of Bengali's 'র', has no 'জ'/'য' 
    distinction, and uses different verb conjugations. Do not default to Bengali script/grammar."
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        return "Sorry, I couldn't process that. Please try again."
        
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    examples_str = "\n".join([f"English: {ex['en']} → Assamese: {ex['as']}" for ex in ASSAMESE_EXAMPLES[:4]])
    
    system_prompt = (
        "You are a healthcare and governance assistant for Northeast India, specifically Assam. "
        "ONLY answer questions about healthcare, medical symptoms, government schemes, or governance procedures. "
        "If the question is about anything else (tourism, entertainment, general knowledge, etc.), "
        "respond ONLY with: 'I can only help with healthcare and governance questions. Please ask about medical concerns or government schemes.' in Assamese. "
        "Do not answer off-topic questions even partially. "
        "Users may mix languages (Hindi, English, Assamese) in their question. Understand the intent regardless of language mixing, and still respond in Assamese. "
        "Respond strictly in Assamese (অসমীয়া), NOT Bengali. Assamese has distinct "
        "features from Bengali — e.g., Assamese uses 'ৰ' (ra) instead of Bengali's 'র', has no 'জ'/'য' "
        "distinction, and uses different verb conjugations. Do not default to Bengali script/grammar. "
        "Keep your answer under 80 words. Do not create long numbered lists or sequences of numbers/dates. Write in plain sentences.\n\n"
        "Here are examples of authentic Assamese (not Bengali):\n"
        f"{examples_str}\n"
        "Now respond to the user's query in this same authentic Assamese style."
    )
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2,
        "max_tokens": 300,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3
    }
    
    try:
        # Calls the API with a timeout of 30 seconds to handle network delays/timeouts
        response = requests.post(url, headers=headers, json=payload, timeout=60.0)
        
        # Raise HTTPError if response was unsuccessful (status code is not 200)
        response.raise_for_status()
        
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = choice["message"]["content"]
                if reply is not None:
                    reply = reply.strip()
                    
                    # Safety check for repetitive words
                    words = reply.split()
                    word_counts = {}
                    has_repetition = False
                    for w in words:
                        w_clean = w.strip('।,.-!?"\'()[]{}')
                        if w_clean:
                            word_counts[w_clean] = word_counts.get(w_clean, 0) + 1
                            if word_counts[w_clean] > 4:
                                has_repetition = True
                                break
                                
                    if has_repetition:
                        # Simple implementation: split into sentences by '।' or '.', keep only first 3 sentences
                        import re
                        sentences = re.split(r'([।.।])', reply)
                        parts = []
                        current = ""
                        for part in sentences:
                            if part in ('।', '.', '।'):
                                parts.append(current + part)
                                current = ""
                            else:
                                current += part
                        if current:
                            parts.append(current)
                            
                        parts = [p.strip() for p in parts if p.strip()]
                        reply = " ".join(parts[:3])
                        
                    return reply.strip()
                    
        raise ValueError("NVIDIA NIM API response does not contain the expected message format.")
        
    except Exception as e:
        print(f"Error calling NVIDIA NIM API: {e}", file=sys.stderr)
        return "Sorry, I couldn't process that. Please try again."
