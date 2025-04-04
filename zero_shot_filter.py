import logging
from transformers import pipeline as hf_pipeline
from typing import Any

# Initialize logger for zero-shot filtering
logger = logging.getLogger(__name__)

try:
    # Load the zero-shot classification model using a robust DeBERTa v3 Large variant.
    classifier: Any = hf_pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
except Exception as e:
    logger.error(f"Error loading zero-shot classification model: {e}")
    classifier = None

def check_zero_shot(prompt: str, threshold: float = 0.5) -> bool:
    """
    Uses zero-shot classification to determine if a prompt is a harmful instruction.
    
    Candidate labels:
      - "instruction to commit violence"
      - "non-harmful query"
    
    For short prompts (<= 5 words), the threshold is raised to reduce overblocking.
    """
    if classifier is None:
        logger.error("Zero-shot classifier is not available.")
        return False

    candidate_labels = ["instruction to commit violence", "non-harmful query"]
    try:
        result = classifier(prompt, candidate_labels)
    except Exception as e:
        logger.error(f"Error during zero-shot classification: {e}")
        return False

    try:
        harmful_score = result["scores"][result["labels"].index("instruction to commit violence")]
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Error extracting harmful score: {e}")
        return False

    # Adjust threshold for short prompts
    if len(prompt.split()) <= 5:
        threshold = 0.8

    logger.info(f"Zero-shot harmful instruction score: {harmful_score:.3f} (Threshold: {threshold})")
    return harmful_score >= threshold
