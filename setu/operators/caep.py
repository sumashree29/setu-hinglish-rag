"""
CAEP — Context-Aware Entity Preservation.
OWNER: R2 | PHASE: 2 (plan §4.2)

Decides, per candidate entity substitution, whether to preserve the original
entity mention or substitute it — via a logistic-regression confidence gate
over [fuzzy_score, embedding_cosine, entity_frequency].
"""
from typing import List, Dict
from sklearn.linear_model import LogisticRegression


def extract_entity_list(corpus_chunks: List[str]) -> List[str]:
    """Running list of proper nouns / scheme names / bank terms from the corpus.
    TODO (R2): simplest version = capitalized multi-word spans + a manual
    supplement list of known scheme/bank names. Alt: spaCy xx_ent_wiki_sm NER
    if the corpus has enough named entities to make it worthwhile (plan alt #3)."""
    raise NotImplementedError("R2: implement entity extraction")


def build_entity_features(candidate: str, entity: str, entity_freq: Dict[str, int]) -> List[float]:
    """
    Returns [fuzzy_score, embedding_cosine, entity_frequency] for one candidate
    substitution pair.
    TODO (R2): fuzzy_score via RapidFuzz, embedding_cosine via the same
    embedding model used elsewhere (import from setu.embeddings.loader).
    """
    raise NotImplementedError("R2: implement feature extraction")


def fit_caep_gate(features: List[List[float]], labels: List[int]) -> LogisticRegression:
    """Fit the preserve-vs-substitute binary classifier.
    TODO (R2): sklearn LogisticRegression, binary cross-entropy loss (default)."""
    raise NotImplementedError("R2: fit logistic regression gate")


def apply_caep(query: str, entities: List[str], gate: LogisticRegression) -> str:
    """Run the corpus entities through the gate and return the query with
    preserved/substituted entities applied.
    TODO (R2): implement end-to-end application."""
    raise NotImplementedError("R2: implement CAEP application")
