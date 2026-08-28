"""
CAEP — Context-Aware Entity Preservation.
OWNER: R2 | PHASE: 2 (plan §4.2)

Decides, per candidate entity substitution, whether to preserve the original
entity mention or substitute it — via a logistic-regression confidence gate
over [fuzzy_score, embedding_cosine, entity_frequency].
"""
import re
from typing import List, Dict, Callable
import numpy as np
from sklearn.linear_model import LogisticRegression
from rapidfuzz import fuzz

STOPWORDS = {
    "the", "a", "an", "and", "of", "in", "on", "at", "to", "for", "with", "by", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
    "this", "that", "these", "those"
}

def extract_entity_list(corpus_chunks: List[str]) -> List[str]:
    """Running list of proper nouns / scheme names / bank terms from the corpus.
    Simplest version = capitalized multi-word spans + a manual supplement list
    of known scheme/bank names. Leading stopwords are stripped.
    """
    entities = set()
    # Match sequences of capitalized words (allowing typical connector words)
    pattern = re.compile(
        r'\b[A-Z][a-zA-Z0-9]*(?:\s+(?:of|and|in|the|for|to|with)\s+[A-Z][a-zA-Z0-9]*|\s+[A-Z][a-zA-Z0-9]*)*\b'
    )
    for chunk in corpus_chunks:
        matches = pattern.findall(chunk)
        for match in matches:
            words = match.split()
            # Strip leading stopwords
            while words and words[0].lower() in STOPWORDS:
                words.pop(0)
            # Strip trailing stopwords
            while words and words[-1].lower() in STOPWORDS:
                words.pop()
            if words:
                entities.add(" ".join(words))
                # Also include individual capitalized words if it's a multi-word phrase
                if len(words) > 1:
                    for w in words:
                        if w.lower() not in STOPWORDS and w[0].isupper():
                            entities.add(w)
    return sorted(list(entities))


def entity_frequencies(corpus_chunks: List[str]) -> Dict[str, int]:
    """Returns the frequency of each extracted entity in the corpus chunks."""
    entities = extract_entity_list(corpus_chunks)
    freqs = {}
    for entity in entities:
        escaped = re.escape(entity)
        pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
        count = 0
        for chunk in corpus_chunks:
            count += len(pattern.findall(chunk))
        freqs[entity] = count
    return freqs


def trigram_cosine(s1: str, s2: str) -> float:
    """Computes trigram cosine similarity between two strings."""
    def get_trigrams(s: str) -> Dict[str, int]:
        s = s.lower()
        t = [s[i:i+3] for i in range(len(s) - 2)]
        if not t:
            t = list(s)
        freq = {}
        for gram in t:
            freq[gram] = freq.get(gram, 0) + 1
        return freq

    f1 = get_trigrams(s1)
    f2 = get_trigrams(s2)
    all_keys = set(f1.keys()).union(set(f2.keys()))
    dot = sum(f1.get(k, 0) * f2.get(k, 0) for k in all_keys)
    norm1 = sum(v * v for v in f1.values()) ** 0.5
    norm2 = sum(v * v for v in f2.values()) ** 0.5
    if norm1 > 0 and norm2 > 0:
        return float(dot / (norm1 * norm2))
    return 0.0


def build_entity_features(candidate: str, entity: str, entity_freq: Dict[str, int], embed_fn: Callable = None) -> List[float]:
    """
    Returns [fuzzy_score, embedding_cosine, entity_frequency] for one candidate
    substitution pair.
    
    fuzzy_score via RapidFuzz, embedding_cosine via the provided embed_fn (falls
    back to a trigram-cosine if embed_fn is None).
    """
    fuzzy_score = float(fuzz.ratio(candidate.lower(), entity.lower()))
    
    if embed_fn is not None:
        try:
            embeddings = embed_fn([candidate, entity])
            vec1 = np.asarray(embeddings[0])
            vec2 = np.asarray(embeddings[1])
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                embedding_cosine = float(dot / (norm1 * norm2))
            else:
                embedding_cosine = 0.0
        except Exception:
            embedding_cosine = trigram_cosine(candidate, entity)
    else:
        embedding_cosine = trigram_cosine(candidate, entity)
        
    freq = float(entity_freq.get(entity, 0))
    return [fuzzy_score, embedding_cosine, freq]


def fit_caep_gate(features: List[List[float]], labels: List[int]) -> LogisticRegression:
    """Fit the preserve-vs-substitute binary classifier.
    LogisticRegression with default binary cross-entropy loss.
    """
    clf = LogisticRegression()
    clf.fit(features, labels)
    return clf


def apply_caep(query: str, entities: List[str], gate: LogisticRegression, entity_freq: Dict[str, int] = None, embed_fn: Callable = None) -> str:
    """Run the corpus entities through the gate and return the query with
    preserved/substituted entities applied.
    """
    if entity_freq is None:
        entity_freq = {e: 1 for e in entities}
        
    words = query.split()
    new_words = []
    for word in words:
        # Separate leading/trailing non-alphanumeric punctuation
        match_left = re.match(r'^([^\w]+)', word)
        match_right = re.search(r'([^\w]+)$', word)
        
        left_punct = match_left.group(1) if match_left else ""
        right_punct = match_right.group(1) if match_right else ""
        
        clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        if not clean_word:
            new_words.append(word)
            continue
            
        best_entity = None
        best_score = -1.0
        for entity in entities:
            # We check the case-insensitive fuzzy ratio
            score = fuzz.ratio(clean_word.lower(), entity.lower())
            if score > best_score:
                best_score = score
                best_entity = entity
                
        if best_entity and best_score > 50:
            features = build_entity_features(clean_word, best_entity, entity_freq, embed_fn=embed_fn)
            pred = gate.predict([features])[0]
            if pred == 1:
                new_words.append(left_punct + best_entity + right_punct)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
            
    return " ".join(new_words)
