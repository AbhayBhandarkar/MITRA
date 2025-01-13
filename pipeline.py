# pipeline.py

import torch
import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from langchain_ollama import OllamaLLM
import re
import json
from collections import OrderedDict

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('safety_guardrail.log'),
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
    """Custom exception for safety violations with detailed reporting."""
    def __init__(self, check_name: str, details: str):
        self.check_name = check_name
        self.details = details
        super().__init__(f"Safety violation in {check_name}: {details}")

class LRUCache(OrderedDict):
    """Thread-safe LRU cache for storing safety check results."""
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
    """Enhanced guardrail system combining multiple safety checks with caching and detailed reporting."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._initialize_models()
        self.cache = LRUCache(maxsize=self.config['cache_size'])
        self.violation_patterns = self._compile_patterns()
        self.context_window: List[Dict] = []
        self.context_window_size = self.config['context_window_size']

    def _load_config(self, config_path: Optional[str]) -> Dict[Any, Any]:
        """Load configuration with fallback to default values."""
        default_config = {
            'cache_size': 1000,
            'context_window_size': 10,
            'thresholds': {
                'toxicity': 0.7,
                'jailbreak': 0.65,
                'semantic_similarity': 0.75,
                'malicious_intent': 0.6
            },
            'models': {
                'toxicity': 's-nlp/roberta_toxicity_classifier',
                'jailbreak': 'madhurjindal/Jailbreak-Detector-Large',
                'embeddings': 'all-MiniLM-L6-v2'
            },
            'harmful_patterns': [
                r"bypass.*security",
                r"hack.*system",
                r"exploit.*vulnerability",
                r"ignore.*instructions",
                r"override.*safety",
            ]
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    return {**default_config, **loaded_config}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        
        return default_config

    def _initialize_models(self) -> None:
        """Initialize all required models with error handling."""
        try:
            self.toxicity_tokenizer = AutoTokenizer.from_pretrained(
                self.config['models']['toxicity']
            )
            self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(
                self.config['models']['toxicity']
            ).eval()

            self.jailbreak_tokenizer = AutoTokenizer.from_pretrained(
                self.config['models']['jailbreak']
            )
            self.jailbreak_model = AutoModelForSequenceClassification.from_pretrained(
                self.config['models']['jailbreak']
            ).eval()

            self.embedding_model = SentenceTransformer(self.config['models']['embeddings'])

            self.llm = OllamaLLM(model="llama3.2")

            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for pattern matching checks."""
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
            details=f"Jailbreak detection score: {jailbreak_score:.3f}"
        )

    async def evaluate_prompt(self, prompt: str) -> Tuple[bool, List[SafetyCheckResult]]:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cached_result = await self.cache.get(prompt_hash)
        if cached_result:
            logger.info("Using cached safety check results")
            return cached_result

        safety_checks = await asyncio.gather(
            self.check_toxicity(prompt),
            self.check_jailbreak(prompt)
        )

        is_safe = all(check.is_safe for check in safety_checks)
        await self.cache.put(prompt_hash, (is_safe, safety_checks))
        
        return is_safe, safety_checks

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        try:
            is_safe, safety_checks = await self.evaluate_prompt(prompt)
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'is_safe': is_safe,
                'checks': [vars(check) for check in safety_checks],
                'status': 'success'
            }
            
            if is_safe:
                try:
                    response = self.llm.invoke(prompt)
                    result['response'] = response
                except Exception as e:
                    logger.error(f"LLM response generation failed: {e}")
                    result['status'] = 'error'
                    result['error'] = str(e)
            else:
                result['status'] = 'blocked'
                violations = [check.check_name for check in safety_checks if not check.is_safe]
                result['error'] = f"Safety checks failed: {', '.join(violations)}."
                
            return result
                
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Instantiate EnhancedGuardrail once at the module level
_guardrail_instance = EnhancedGuardrail()

async def pipeline(prompt: str) -> Dict[str, Any]:
    return await _guardrail_instance.process_prompt(prompt)
