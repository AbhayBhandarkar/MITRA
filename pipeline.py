import torch
import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from collections import OrderedDict
import spacy
from langchain_ollama import OllamaLLM

from zero_shot_filter import check_zero_shot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('logs/safety_guardrail.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SafetyCheckResult:
    check_name: str
    is_safe: bool
    confidence: float
    threshold: float
    details: str
    timestamp: datetime = datetime.now()

class SafetyViolation(Exception):
    """Custom exception for safety violations."""
    def __init__(self, check_name: str, details: str):
        self.check_name = check_name
        self.details = details
        super().__init__(f"Safety violation in {check_name}: {details}")

class LRUCache(OrderedDict):
    """Thread-safe LRU cache."""
    def __init__(self, maxsize: int = 1000):
        super().__init__()
        self.maxsize = maxsize
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self:
                return None
            value = self[key]
            self.move_to_end(key)
            return value

    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self:
                self.move_to_end(key)
            self[key] = value
            if len(self) > self.maxsize:
                self.popitem(last=False)

class EnhancedGuardrail:
    """Enhanced guardrail system for safety checks."""
    def __init__(self):
        self.config = self._load_config()
        self._initialize_models()
        self.cache = LRUCache(maxsize=self.config['cache_size'])
        self.violation_patterns = self._compile_patterns()
        self.nlp = spacy.load("en_core_web_sm")
        self.llm = OllamaLLM(model="qwen2.5:0.5b")

    def _load_config(self) -> Dict[str, Any]:
        return {
            'cache_size': 1000,
            'thresholds': {
                'toxicity': 0.5,
                'jailbreak': 0.8,
                'zero_shot': 0.7,  # threshold for the zero-shot classifier
                'combined': 0.6
            },
            'models': {
                'toxicity': 's-nlp/roberta_toxicity_classifier',
                'jailbreak': 'madhurjindal/Jailbreak-Detector-Large'
            },
            'harmful_patterns': [
                r"\b(make|build|manufacture)\b.*\b(bomb)\b"
            ]
        }

    def _initialize_models(self) -> None:
        try:
            self.toxicity_tokenizer = AutoTokenizer.from_pretrained(self.config['models']['toxicity'])
            self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(self.config['models']['toxicity']).eval()
            self.jailbreak_tokenizer = AutoTokenizer.from_pretrained(self.config['models']['jailbreak'])
            self.jailbreak_model = AutoModelForSequenceClassification.from_pretrained(self.config['models']['jailbreak']).eval()
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _compile_patterns(self) -> List[re.Pattern]:
        return [re.compile(pattern, re.IGNORECASE) for pattern in self.config['harmful_patterns']]

    async def check_toxicity(self, text: str) -> SafetyCheckResult:
        inputs = self.toxicity_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.toxicity_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            toxicity_score = scores[0][1].item()
        return SafetyCheckResult(
            check_name="Toxicity",
            is_safe=toxicity_score < self.config['thresholds']['toxicity'],
            confidence=toxicity_score,
            threshold=self.config['thresholds']['toxicity'],
            details=f"Toxicity score: {toxicity_score:.3f}"
        )

    async def check_jailbreak(self, text: str) -> SafetyCheckResult:
        inputs = self.jailbreak_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.jailbreak_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            jailbreak_score = scores[0][1].item()
        return SafetyCheckResult(
            check_name="Jailbreak",
            is_safe=jailbreak_score < self.config['thresholds']['jailbreak'],
            confidence=jailbreak_score,
            threshold=self.config['thresholds']['jailbreak'],
            details=f"Jailbreak score: {jailbreak_score:.3f}"
        )

    async def check_patterns(self, text: str) -> SafetyCheckResult:
        matches = [pattern.search(text) for pattern in self.violation_patterns]
        found_patterns = [match.group() for match in matches if match]
        is_safe = not found_patterns
        return SafetyCheckResult(
            check_name="Harmful Patterns",
            is_safe=is_safe,
            confidence=0.0 if not is_safe else 1.0,
            threshold=0.0,
            details=f"Matched patterns: {found_patterns}" if found_patterns else "No harmful patterns detected"
        )

    async def evaluate_prompt(self, prompt: str) -> Tuple[bool, List[SafetyCheckResult]]:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cached_result = await self.cache.get(prompt_hash)
        if cached_result:
            logger.info("Using cached safety check results")
            return cached_result

        safety_checks = await asyncio.gather(
            self.check_toxicity(prompt),
            self.check_jailbreak(prompt),
            self.check_patterns(prompt)
        )

        is_safe = all(check.is_safe for check in safety_checks)
        await self.cache.put(prompt_hash, (is_safe, safety_checks))
        return is_safe, safety_checks

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        try:
            # Use the zero-shot classifier to preemptively block harmful instructions.
            if check_zero_shot(prompt, threshold=self.config['thresholds']['zero_shot']):
                logger.info("Zero-shot filter flagged the prompt as harmful.")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'prompt': prompt,
                    'is_safe': False,
                    'checks': [{
                        "check_name": "Zero-shot Classification",
                        "is_safe": False,
                        "confidence": None,
                        "threshold": self.config['thresholds']['zero_shot'],
                        "details": "Prompt classified as an instruction to commit violence."
                    }],
                    'status': 'blocked',
                    'error': "Blocked due to classification as a harmful instruction."
                }

            # Proceed with additional safety checks if the zero-shot filter passes.
            is_safe, safety_checks = await self.evaluate_prompt(prompt)
            result = {
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'is_safe': is_safe,
                'checks': [vars(check) for check in safety_checks]
            }

            if not is_safe:
                result['status'] = 'blocked'
                violations = [check.check_name for check in safety_checks if not check.is_safe]
                result['error'] = f"Blocked due to: {', '.join(violations)}"
            else:
                result['status'] = 'allowed'
                try:
                    response = self.llm.invoke(prompt)
                    result['response'] = response
                except Exception as e:
                    logger.error(f"Failed to generate response: {e}")
                    result['response'] = "Error generating response."

            return result

        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            return {'status': 'error', 'error': str(e)}

# Instantiate the guardrail once so that models are loaded only a single time.
_guardrail_instance = EnhancedGuardrail()

async def pipeline(prompt: str) -> Dict[str, Any]:
    return await _guardrail_instance.process_prompt(prompt)
