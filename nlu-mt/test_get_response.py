import os
import sys
from get_response import get_response

# Reconfigure stdout to use UTF-8 encoding on Windows to prevent UnicodeEncodeError when printing Assamese characters
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main():
    # Log details about the API Key presence (safely masked)
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "..."
        print(f"Using NVIDIA_API_KEY: {masked_key}")
    else:
        print("WARNING: NVIDIA_API_KEY environment variable is not set. API calls will fail.")

    # 3-4 sample healthcare/governance questions in English, plus off-topic and code-mixed tests
    questions = [
        "What are the symptoms of malaria and how can I get tested in Assam?",
        "How can I apply for the Orunodoi scheme in Assam?",
        "What should I do if I have high fever and body ache?",
        "Can you tell me about the best tourism spots in Northeast India?", # Unrelated query
        "What's the weather like today?", # Off-topic query (should get redirect message)
        "Mera fever hai, ki koru?" # Code-mixed query (should get relevant healthcare response in Assamese)
    ]
    
    print("\n--- Testing get_response() ---")
    for q in questions:
        print(f"\nEnglish Question: {q}")
        print("Fetching response...")
        reply = get_response(q)
        print(f"Assamese Answer: {reply}")

if __name__ == "__main__":
    main()
