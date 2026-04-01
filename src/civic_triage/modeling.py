"""TF-IDF + multinomial logistic baseline and fine-tuned variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainedRouter:
    pipeline: Pipeline
    label_encoder: LabelEncoder

    def predict(self, texts: list[str]) -> np.ndarray:
        encoded = self.pipeline.predict(texts)
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)


def build_router(
    max_features: int = 12000,
    C: float = 1.0,
    seed: int = 42,
) -> tuple[Pipeline, LabelEncoder]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
    )
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=C,
        random_state=seed,
    )
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    encoder = LabelEncoder()
    return pipe, encoder


def fit_sft(
    texts: list[str],
    labels: list[str],
    seed: int = 42,
) -> TrainedRouter:
    pipe, enc = build_router(seed=seed)
    y = enc.fit_transform(labels)
    pipe.fit(texts, y)
    return TrainedRouter(pipeline=pipe, label_encoder=enc)


def metrics_for(
    router: TrainedRouter,
    texts: list[str],
    labels: list[str],
) -> dict[str, float]:
    pred = router.predict(texts)
    y_true = np.array(labels)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro", zero_division=0)
        ),
    }


def apply_preference_alignment(
    base_texts: list[str],
    base_labels: list[str],
    pairs: list[tuple[str, str, str]],
    oversample_chosen: int = 3,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Approximate preference optimization by duplicating chosen-label examples
    and adding contrastive emphasis for reviewer corrections.
    """
    rng = np.random.default_rng(seed)
    texts = list(base_texts)
    labels = list(base_labels)
    for text, chosen, rejected in pairs:
        for _ in range(oversample_chosen):
            texts.append(text)
            labels.append(chosen)
        # Light negative signal: a single counter-example label (rejected) with downweight
        # represented by one extra sample with different label is too harsh for linear model;
        # instead we add a small perturbed duplicate toward rejected for regularization balance.
        if rng.random() < 0.15:
            texts.append(text + " [reviewer_note: not " + rejected + "]")
            labels.append(chosen)
    return texts, labels
