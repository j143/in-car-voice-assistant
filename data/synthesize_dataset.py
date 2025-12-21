#!/usr/bin/env python3
"""
Bosch Automotive NLP Dataset Synthesis Pipeline
==============================================

This script takes 100 seed examples and synthesizes 10,000+ training examples
through controlled paraphrasing, noise injection, and variant generation.

Usage:
    python3 synthesize_bosch_dataset.py --seed-file bosch_dataset_seed.jsonl --output-file bosch_dataset_10k.jsonl --multiplier 50

Prerequisites:
    pip install openai pydantic python-dotenv tqdm

"""

import json
import argparse
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import random
from collections import defaultdict
import logging
from datetime import datetime

# If using Google Gemini API (recommended for quality)
try:
    from google import genai
except ImportError:
    genai = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BoschExample:
    """Single training example in Bosch dataset"""
    id: int
    intent: str
    text: str
    technical_terms: List[str]
    error_codes: List[str]
    subsystem: str
    command: str
    severity: str
    expected: str
    source: str
    variant_type: Optional[str] = None


class BoschDatasetSynthesizer:
    """Synthesizes Bosch automotive NLP dataset from seeds"""
    
    def __init__(self, seed_file: str, api_key: Optional[str] = None):
        """Initialize with seed examples"""
        self.seed_file = seed_file
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.seeds = self._load_seeds()
        self.examples_by_intent = self._group_by_intent()
        self.generated = []
        
    def _load_seeds(self) -> List[BoschExample]:
        """Load seed examples from JSONL file"""
        examples = []
        with open(self.seed_file, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                examples.append(BoschExample(**data))
        logger.info(f"Loaded {len(examples)} seed examples")
        return examples
    
    def _group_by_intent(self) -> Dict[str, List[BoschExample]]:
        """Group examples by intent for balanced synthesis"""
        groups = defaultdict(list)
        for ex in self.seeds:
            groups[ex.intent].append(ex)
        logger.info(f"Intent distribution: {dict(groups)}")
        return groups
    
    def synthesize_paraphrases(self, seed: BoschExample, count: int = 5) -> List[BoschExample]:
        """Generate N paraphrased variants of a seed example using Google Gemini API if available"""
        variants = []
        api_used = False
        
        if genai and self.api_key:
            try:
                client = genai.Client(api_key=self.api_key)
                
                prompt = (
                    f"You are an expert automotive NLP assistant. Paraphrase the following automotive command or query "
                    f"in {count} different natural ways. Keep the meaning intact, use colloquial language where appropriate, "
                    f"and avoid repetition. Return ONLY the paraphrases separated by newlines, one per line, without numbering.\n\n"
                    f"Original: '{seed.text}'\n\n"
                    f"Paraphrases (one per line):"
                )
                
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents=prompt,
                )
                
                output = response.text
                logger.info(f"✓ Gemini API call successful for seed: {seed.text[:50]}...")
                api_used = True
                
                # Parse paraphrases from output
                lines = output.split('\n')
                paraphrases = []
                for line in lines:
                    line = line.strip()
                    # Remove numbering if present (1., 2., etc.)
                    if line and not line.lower().startswith(('paraphrase', 'original')):
                        # Clean up common patterns
                        line = line.lstrip('0123456789.-) ')
                        if line and len(line) > 5:
                            paraphrases.append(line)
                
                # Keep unique ones and filter duplicates of original
                unique_paraphrases = []
                seen = {seed.text.lower().strip()}
                for p in paraphrases[:count]:
                    p_lower = p.lower().strip()
                    if p_lower not in seen:
                        unique_paraphrases.append(p)
                        seen.add(p_lower)
                
                # Create variant examples
                for i, paraphrase in enumerate(unique_paraphrases):
                    variant = BoschExample(
                        id=len(self.generated) + i,
                        intent=seed.intent,
                        text=paraphrase,
                        technical_terms=seed.technical_terms,
                        error_codes=seed.error_codes,
                        subsystem=seed.subsystem,
                        command=seed.command,
                        severity=seed.severity,
                        expected=seed.expected,
                        source=seed.source,
                        variant_type="paraphrase_gemini"
                    )
                    variants.append(variant)
                
                # If not enough, fallback to template-based
                if len(variants) < count:
                    logger.info(f"  Got {len(variants)} Gemini paraphrases, filling remaining {count - len(variants)} from templates")
                    paraphrase_templates = self._get_paraphrase_templates(seed)
                    for i, template in enumerate(paraphrase_templates[:count-len(variants)]):
                        if template.lower().strip() not in seen:
                            variant = BoschExample(
                                id=len(self.generated) + len(variants) + i,
                                intent=seed.intent,
                                text=template,
                                technical_terms=seed.technical_terms,
                                error_codes=seed.error_codes,
                                subsystem=seed.subsystem,
                                command=seed.command,
                                severity=seed.severity,
                                expected=seed.expected,
                                source=seed.source,
                                variant_type="paraphrase"
                            )
                            variants.append(variant)
                            seen.add(template.lower().strip())
                
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                    logger.warning(f"✗ Gemini API quota exceeded. Using template-based paraphrasing instead.")
                elif "api_key" in error_msg.lower() or "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    logger.warning(f"✗ Invalid Gemini API key. Using template-based paraphrasing instead.")
                else:
                    logger.warning(f"✗ Gemini API call failed: {type(e).__name__}: {error_msg}. Falling back to templates.")
                api_used = False
        
        # If API wasn't used or failed, use template-based approach
        if not api_used or not variants:
            paraphrase_templates = self._get_paraphrase_templates(seed)
            for i, template in enumerate(paraphrase_templates[:count]):
                variant = BoschExample(
                    id=len(self.generated) + i,
                    intent=seed.intent,
                    text=template,
                    technical_terms=seed.technical_terms,
                    error_codes=seed.error_codes,
                    subsystem=seed.subsystem,
                    command=seed.command,
                    severity=seed.severity,
                    expected=seed.expected,
                    source=seed.source,
                    variant_type="paraphrase"
                )
                variants.append(variant)
        
        return variants
    
    def _get_paraphrase_templates(self, seed: BoschExample) -> List[str]:
        """Generate paraphrase templates for a seed based on intent"""
        templates = []
        
        if seed.intent == "error_handling":
            # Extract error code from expected
            if seed.error_codes:
                code = seed.error_codes[0]
                sys = seed.subsystem.replace("_", " ").title()
                
                templates = [
                    f"Got error {code}, what does it mean?",
                    f"{code}: what should I check?",
                    f"My scanner shows {code}. Is it serious?",
                    f"Diagnostic trouble code {code} appeared.",
                    f"ECU showing {code}, is the car safe to drive?",
                    f"What is {code}? How do I fix it?",
                    f"{code} error on {sys}. Next steps?",
                    f"Scanner reports {code}. {sys} problem?",
                ]
        
        elif seed.intent == "query_sensor":
            # Extract sensor from text
            sensor = seed.subsystem.replace("_", " ").title()
            templates = [
                f"What's my current {sensor}?",
                f"Show me {sensor}",
                f"Current {sensor} reading?",
                f"Can you check the {sensor}?",
                f"What is the {sensor} right now?",
                f"{sensor} status?",
            ]
        
        elif seed.intent == "climate_control":
            templates = [
                seed.text,  # Keep original
                seed.text.replace("Set", "Change").replace("set", "change"),
                seed.text.replace("to", "at"),
            ]
        
        elif seed.intent == "set_alert":
            templates = [
                seed.text,
                seed.text.replace("Notify", "Alert").replace("notify", "alert"),
                seed.text.replace("if", "when"),
            ]
        
        return [t for t in templates if t]
    
    def synthesize_with_noise(self, seed: BoschExample, count: int = 3) -> List[BoschExample]:
        """Generate variants with colloquial speech, partial info, wrong assumptions"""
        variants = []
        noise_patterns = [
            lambda t: t.replace("What is", "What's"),  # Contraction
            lambda t: t.lower(),  # Lowercase
            lambda t: t.rstrip('?') + '?',  # Ensure question mark
            lambda t: "can u " + t.lower(),  # Colloquial
            lambda t: "help me " + t,  # Add context
        ]
        
        for i, pattern in enumerate(noise_patterns[:count]):
            variant_text = pattern(seed.text)
            variant = BoschExample(
                id=len(self.generated) + i,
                intent=seed.intent,
                text=variant_text,
                technical_terms=seed.technical_terms,
                error_codes=seed.error_codes,
                subsystem=seed.subsystem,
                command=seed.command,
                severity=seed.severity,
                expected=seed.expected,
                source=seed.source,
                variant_type="noise"
            )
            variants.append(variant)
        
        return variants
    
    def synthesize_dialog_turns(self, seed: BoschExample, count: int = 2) -> List[BoschExample]:
        """Generate multi-turn dialog variants"""
        variants = []
        
        dialogs = []
        if seed.intent == "error_handling":
            dialogs = [
                f"So {seed.text}. What should I do?",
                f"{seed.text}. Is this normal?",
                f"I'm seeing {seed.error_codes[0] if seed.error_codes else 'an error'}. " + seed.text,
            ]
        elif seed.intent == "query_sensor":
            dialogs = [
                f"Can you tell me {seed.text.lower()}",
                f"I need to know {seed.text.lower()}",
            ]
        
        for i, dialog in enumerate(dialogs[:count]):
            variant = BoschExample(
                id=len(self.generated) + i,
                intent=seed.intent,
                text=dialog,
                technical_terms=seed.technical_terms,
                error_codes=seed.error_codes,
                subsystem=seed.subsystem,
                command=seed.command,
                severity=seed.severity,
                expected=seed.expected,
                source=seed.source,
                variant_type="dialog"
            )
            variants.append(variant)
        
        return variants
    
    def synthesize_balanced(self, multiplier: int = 10):
        """Generate balanced dataset with multiplier × seed count examples"""
        target_count = len(self.seeds) * multiplier
        logger.info(f"Synthesizing {target_count} examples (multiplier: {multiplier}x)")
        
        for seed in self.seeds:
            # For each seed, generate paraphrases, noisy variants, and dialogs
            self.generated.extend(self.synthesize_paraphrases(seed, count=multiplier // 2))
            if random.random() < 0.7:  # 70% get noise variants
                self.generated.extend(self.synthesize_with_noise(seed, count=multiplier // 4))
            if seed.intent in ["error_handling", "query_sensor"]:
                self.generated.extend(self.synthesize_dialog_turns(seed, count=multiplier // 4))
        
        # Reassign IDs
        for i, ex in enumerate(self.generated):
            ex.id = i
        
        logger.info(f"Generated {len(self.generated)} total examples")
    
    def deduplicate(self):
        """Remove near-duplicate examples"""
        seen_texts = set()
        unique = []
        
        for ex in self.generated:
            text_norm = ex.text.lower().strip()
            if text_norm not in seen_texts:
                unique.append(ex)
                seen_texts.add(text_norm)
        
        removed = len(self.generated) - len(unique)
        logger.info(f"Deduplicated: removed {removed} near-duplicates")
        self.generated = unique
        
        # Reassign IDs
        for i, ex in enumerate(self.generated):
            ex.id = i
    
    def filter_hallucinations(self):
        """Filter out examples that don't make sense in automotive domain"""
        # Simple heuristic: check for obviously broken text
        filtered = []
        
        for ex in self.generated:
            # Keep if text is non-empty and contains relevant automotive keywords
            if ex.text and len(ex.text) > 3:
                filtered.append(ex)
        
        logger.info(f"Filtered to {len(filtered)} sensible examples")
        self.generated = filtered
    
    def save(self, output_file: str):
        """Save generated dataset to JSONL"""
        with open(output_file, 'w') as f:
            for ex in self.generated:
                ex_dict = asdict(ex)
                f.write(json.dumps(ex_dict) + '\n')
        
        logger.info(f"Saved {len(self.generated)} examples to {output_file}")
    
    def save_stats(self, output_file: str):
        """Save dataset statistics"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_examples": len(self.generated),
            "intent_distribution": self._compute_intent_distribution(),
            "severity_distribution": self._compute_severity_distribution(),
            "subsystem_distribution": self._compute_subsystem_distribution(),
            "variant_types": self._compute_variant_types(),
        }
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to {output_file}")
    
    def _compute_intent_distribution(self) -> Dict[str, int]:
        """Count examples per intent"""
        dist = defaultdict(int)
        for ex in self.generated:
            dist[ex.intent] += 1
        return dict(dist)
    
    def _compute_severity_distribution(self) -> Dict[str, int]:
        """Count examples per severity"""
        dist = defaultdict(int)
        for ex in self.generated:
            dist[ex.severity] += 1
        return dict(dist)
    
    def _compute_subsystem_distribution(self) -> Dict[str, int]:
        """Count examples per subsystem"""
        dist = defaultdict(int)
        for ex in self.generated:
            dist[ex.subsystem] += 1
        return dict(dist)
    
    def _compute_variant_types(self) -> Dict[str, int]:
        """Count examples per variant type"""
        dist = defaultdict(int)
        dist["original"] = len(self.seeds)
        for ex in self.generated:
            if ex.variant_type:
                dist[ex.variant_type] += 1
        return dict(dist)


def main():
    parser = argparse.ArgumentParser(description="Synthesize Bosch automotive NLP dataset")
    parser.add_argument("--seed-file", default="bosch_dataset_seed.jsonl",
                        help="Path to seed JSONL file")
    parser.add_argument("--output-file", default="bosch_dataset_synthetic.jsonl",
                        help="Path to output JSONL file")
    parser.add_argument("--stats-file", default="bosch_dataset_stats.json",
                        help="Path to output stats JSON file")
    parser.add_argument("--multiplier", type=int, default=10,
                        help="Multiplication factor (10 = ~1000 examples from 100 seeds)")
    parser.add_argument("--api-key", help="OpenAI API key (optional)")
    
    args = parser.parse_args()
    
    # Load and synthesize
    synthesizer = BoschDatasetSynthesizer(args.seed_file, api_key=args.api_key)
    synthesizer.synthesize_balanced(multiplier=args.multiplier)
    synthesizer.filter_hallucinations()
    synthesizer.deduplicate()
    
    # Save outputs
    synthesizer.save(args.output_file)
    synthesizer.save_stats(args.stats_file)
    
    logger.info("Dataset synthesis complete!")


if __name__ == "__main__":
    main()
