import re
import json
import sys
from datasets import load_dataset

# Reconfigure stdout to use UTF-8 encoding on Windows to prevent UnicodeEncodeError when printing Assamese characters
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def is_clean_conversational(en, as_str):
    # Skip if length is > 15 words or < 3 words in either language
    if len(en.split()) > 15 or len(as_str.split()) > 15:
        return False
    if len(en.split()) < 3 or len(as_str.split()) < 3:
        return False
        
    # Skip if contains digits/numbers
    if any(c.isdigit() for c in en) or any(c.isdigit() for c in as_str):
        return False
        
    # Skip if contains weird symbols
    weird_chars = re.compile(r'[@#$%^&*_+={}\[\]<>/\\|`~]')
    if weird_chars.search(en) or weird_chars.search(as_str):
        return False
        
    # Priority for conversational structures in English
    conversational_starts = (
        "what", "how", "where", "why", "who", "when", "please", "do", "you", "i", "we", "he", "she", 
        "they", "this", "that", "it", "is", "are", "have", "has", "can", "could", "should", "would",
        "go", "come", "take", "drink", "rest", "help", "need", "give", "tell", "say", "know", "see"
    )
    en_lower = en.lower().strip()
    if not en_lower.startswith(conversational_starts):
        return False
        
    return True

def main():
    print("Loading ai4bharat/samanantar for Assamese...")
    # Load dataset
    dataset = load_dataset("ai4bharat/samanantar", "as", split="train")
    
    print("Filtering conversational examples...")
    examples = []
    seen_en = set()
    
    for row in dataset:
        en = row["src"]
        as_str = row["tgt"]
        
        if en in seen_en:
            continue
            
        if is_clean_conversational(en, as_str):
            examples.append({"en": en.strip(), "as": as_str.strip()})
            seen_en.add(en)
            if len(examples) >= 15:
                break
                
    # Pick top 10 clean examples
    final_examples = examples[:10]
    
    # Print the formatted Python list
    print("\nCopy-pasteable Python list:")
    formatted_list = json.dumps(final_examples, ensure_ascii=False, indent=4)
    print(formatted_list)
    
    # Save to /nlu-mt/assamese_examples.py
    output_path = "assamese_examples.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated conversational Assamese-English sentence pairs\n")
        f.write(f"ASSAMESE_EXAMPLES = {formatted_list}\n")
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
