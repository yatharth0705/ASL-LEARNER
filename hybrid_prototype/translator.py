import os
import re
import time
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GlossaBARTTranslator:
    def __init__(self, model_dir: str = "feasibility_workspace/stage2_bart"):
        model_dir = os.path.abspath(model_dir)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Stage 2 BART directory not found at {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()

    def normalize_gloss_token(self, token: str) -> str:
        """Normalize ASL gloss notation for optimal BART syntactic comprehension."""
        t = token.strip().upper()
        # Remove loan sign prefix #
        if t.startswith("#"):
            t = t[1:]
        # Deictic index pronouns
        if t in ("IX", "IX-ME", "IX-1P"):
            return "I"
        if t in ("IX-YOU", "IX-2P"):
            return "YOU"
        if t in ("IX-HE", "IX-SHE", "IX-IT", "IX-3P"):
            return "HE"
        # Compound hyphens
        t = t.replace("-", " ")
        return t

    def format_gloss_sequence(self, glosses: List[str]) -> str:
        normalized = [self.normalize_gloss_token(g) for g in glosses if g.strip()]
        return " ".join(normalized)

    def clean_output_text(self, text: str) -> str:
        """Clean tokenization spacing artifacts like 'word .' -> 'word.'"""
        cleaned = text.strip()
        cleaned = re.sub(r'\s+([.,!?;:\'])', r'\1', cleaned)
        cleaned = re.sub(r'([\(])\s+', r'\1', cleaned)
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def translate(
        self,
        glosses: List[str],
        use_beam_search: bool = False,
        num_beams: int = 4,
        max_length: int = 64
    ) -> Dict[str, Any]:
        """
        Translates an ordered gloss sequence into fluent English.
        Default is greedy search (num_beams=1) for minimal latency.
        """
        if not glosses:
            return {
                "input_glosses": [],
                "normalized_prompt": "",
                "english_translation": "",
                "latency_ms": 0.0,
                "strategy": "greedy",
                "tokens_generated": 0
            }

        prompt = self.format_gloss_sequence(glosses)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        beams = num_beams if use_beam_search else 1
        strategy = f"beam({beams})" if use_beam_search else "greedy"

        t_start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=beams,
                early_stopping=True if use_beam_search else False
            )
        latency = (time.perf_counter() - t_start) * 1000.0

        raw_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cleaned_text = self.clean_output_text(raw_text)

        return {
            "input_glosses": list(glosses),
            "normalized_prompt": prompt,
            "english_translation": cleaned_text,
            "raw_output": raw_text,
            "latency_ms": latency,
            "strategy": strategy,
            "tokens_generated": len(output_ids[0])
        }

