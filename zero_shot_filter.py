import logging
from transformers import pipeline as hf_pipeline

# Load the zero-shot classification model using a robust DeBERTa v3 Large variant.
classifier = hf_pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

def check_zero_shot(prompt: str, threshold: float = 0.5) -> bool:
    """
    Uses zero-shot classification to determine if a prompt is a harmful instruction.
    The threshold is dynamically adjusted based on prompt length.
    Candidate labels:
      - "instruction to commit violence"
      - "non-harmful query"
    """
    candidate_labels = ["instruction to commit violence", "non-harmful query"]
    result = classifier(prompt, candidate_labels)
    harmful_score = result["scores"][result["labels"].index("instruction to commit violence")]
    
    # Adjust threshold for shorter prompts to reduce overblocking.
    words = prompt.split()
    if len(words) <= 5:
        threshold = 0.8

    logging.info(f"Zero-shot harmful instruction score: {harmful_score:.3f} (Threshold: {threshold})")
    return harmful_score >= threshold
