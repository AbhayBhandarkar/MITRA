import logging
from transformers import pipeline as hf_pipeline

classifier = hf_pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

def check_zero_shot(prompt: str, threshold: float = 0.7) -> bool:
    """
    Uses zero-shot classification to determine if a prompt is a harmful instruction.
    Candidate labels:
      - "instruction to commit violence" 
      - "non-harmful query"
    If the score for "instruction to commit violence" meets or exceeds the threshold,
    we consider the prompt harmful.
    """
    candidate_labels = ["instruction to commit violence", "non-harmful query"]
    result = classifier(prompt, candidate_labels)
    harmful_score = result["scores"][result["labels"].index("instruction to commit violence")]
    logging.info(f"Zero-shot harmful instruction score: {harmful_score:.3f}")
    return harmful_score >= threshold
