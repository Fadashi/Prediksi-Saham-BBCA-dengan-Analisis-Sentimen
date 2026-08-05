"""Modul pelabelan sentimen berbasis IndoBERT Transformer (indobenchmark/indobert-base-p1).

Mendukung batch processing cepat menggunakan PyTorch & Transformers pipeline.
"""

from typing import List, Dict, Union
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class IndoBERTScorer:
    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[IndoBERT] Loading model: {model_name} (device={self.device})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            truncation=True,
            max_length=512,
        )
        
        self.label_map = {
            "LABEL_0": 1,   # Positif
            "LABEL_1": 0,   # Netral
            "LABEL_2": -1   # Negatif
        }

    def score_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Union[int, float]]]:
        """Memprediksi label (-1, 0, 1) dan skor probabilitas untuk sekelompok teks."""
        results = []
        cleaned_texts = [str(t)[:512] if t and str(t).strip() else "netral" for t in texts]
        
        # Predict using pipeline batching
        predictions = self.pipeline(cleaned_texts, batch_size=batch_size)
        
        for pred in predictions:
            raw_label = pred["label"]
            confidence = float(pred["score"])
            numeric_label = self.label_map.get(raw_label, 0)
            
            # Continuous score: label * confidence (scale [-1.0, 1.0])
            continuous_score = numeric_label * confidence
            
            results.append({
                "sentiment_label": numeric_label,
                "sentiment_score": round(continuous_score, 4),
                "confidence": round(confidence, 4)
            })
            
        return results


def process_indobert_for_dataframe(df: pd.DataFrame, text_column: str = "content", batch_size: int = 64) -> pd.DataFrame:
    """Menambahkan kolom sentiment_score, sentiment_label, dan confidence ke DataFrame."""
    df_result = df.copy()
    scorer = IndoBERTScorer()
    
    texts = df_result[text_column].astype(str).tolist()
    print(f"[IndoBERT] Scoring {len(texts)} postingan (batch_size={batch_size})...")
    
    scored_results = scorer.score_batch(texts, batch_size=batch_size)
    
    df_result["sentiment_label"] = [r["sentiment_label"] for r in scored_results]
    df_result["sentiment_score"] = [r["sentiment_score"] for r in scored_results]
    df_result["confidence"] = [r["confidence"] for r in scored_results]
    
    return df_result
