import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chunks_file = ROOT / "data" / "processed" / "corpus_chunks_v2.jsonl"
chunks = [json.loads(l) for l in open(chunks_file, encoding='utf-8')]
c_chunks = [c for c in chunks if c['chunk_id'].startswith('C')]

def extract_q(text):
    m = re.match(r'Q:\s*(.*?)\nA:', text, re.DOTALL)
    if not m: return None
    q = m.group(1).strip()
    # compound questions: take only the FIRST sentence (up to first '?')
    parts = q.split('?')
    return parts[0].strip() + '?' if parts else q

def strip_numbering(q):
    return re.sub(r'^(Q\.?\s*)?[\divxlIVXL]+[\.\)]\s*', '', q, flags=re.IGNORECASE).strip()

def clean_topic(topic):
    topic = topic.strip()
    # strip leading articles
    topic = re.sub(r'^(a|an|the)\s+', '', topic, flags=re.IGNORECASE)
    # strip leading pronoun+auxiliary fragments left over from clause-splitting ("i track...", "one's")
    topic = re.sub(r"^(i|one'?s?)\s+", '', topic, flags=re.IGNORECASE)
    # collapse " is " / " are " / " be " that read awkwardly mid-clause when frame already supplies the modal
    # (light touch: only strip a leading "is/are" if topic starts with it, mid-clause ones are usually fine)
    topic = re.sub(r'^(is|are)\s+', '', topic, flags=re.IGNORECASE)
    return topic.strip()

def is_structural_header(q):
    junk_patterns = [
        r'^(An?\s+)?illustration:?$', r'^table\b', r'^figure\b', r'^amendments?\b',
        r'^annex', r'^appendix', r'^part\s+[ivxlc]+', r'^chapter\b',
    ]
    for p in junk_patterns:
        if re.match(p, q, re.IGNORECASE):
            return True
    words = q.split()
    has_qword = bool(re.match(r'^(what|who|how|can|is|are|whether|when|should|does|do|will|which)\b', q, re.IGNORECASE))
    if not has_qword and len(words) <= 5:
        return True
    return False

def build_query(q_raw):
    q = strip_numbering(q_raw)
    q = q.rstrip('?.').strip()
    ql = q.lower()

    m = re.match(r'how much (of )?(.*)', ql)
    if m:
        topic = clean_topic(re.sub(r"^(one'?s?)\s+", '', m.group(2)))
        return f"{topic} kitna/kitni hota hai aur kya limit hai"

    m = re.match(r'how many (.*)', ql)
    if m:
        return f"{clean_topic(m.group(1))} kitni hoti hain"

    m = re.match(r'how (will|does|do|can|is|are) (.*)', ql)
    if m:
        topic = clean_topic(m.group(2))
        return f"{topic} kaise hota hai, iska process kya hai"

    m = re.match(r'(whether|can|is there|are there|should|is|are) (.*)', ql)
    if m:
        topic = clean_topic(m.group(2))
        return f"kya {topic} ho sakta hai ya nahi"

    m = re.match(r'(what if|when should|when did|when) (.*)', ql)
    if m:
        topic = clean_topic(m.group(2))
        return f"agar {topic} toh kya karna chahiye, kya procedure hai"

    m = re.match(r'what is the (procedure|process|mandate|time ?frame) for (.*)', ql)
    if m:
        kind, topic = m.group(1), clean_topic(m.group(2))
        return f"{topic} ke liye {kind} kya hai"

    m = re.match(r'what (is|are) (meant by )?(.*)', ql)
    if m:
        topic = clean_topic(m.group(3))
        return f"{topic} kya hota hai"

    m = re.match(r'who (is|are) (a |an )?(.*)', ql)
    if m:
        topic = clean_topic(m.group(3))
        return f"{topic} kaun hota hai"

    m = re.match(r'what does (.*?) (mean|insure|cover)', ql)
    if m:
        topic, verb = clean_topic(m.group(1)), m.group(2)
        return f"{topic} {verb} kya karta hai"

    m = re.match(r'who can (.*)', ql)
    if m:
        topic = clean_topic(m.group(1))
        return f"{topic} ke liye kaun eligible hai, kya documents chahiye"

    m = re.match(r'who (.*)', ql)
    if m:
        topic = clean_topic(m.group(1))
        return f"{topic} kaun hota hai"

    return f"{clean_topic(q)} - iske baare mein jaankari chahiye"

if __name__ == "__main__":
    results = []
    discarded = []
    for c in c_chunks:
        q_raw = extract_q(c['text'])
        if not q_raw or is_structural_header(strip_numbering(q_raw)):
            discarded.append(c['chunk_id'])
            continue
        generated = build_query(q_raw)
        # discard if topic ended up too short/empty after cleaning (signals extraction failure)
        if len(generated.split()) < 4:
            discarded.append(c['chunk_id'])
            continue
        results.append({
            'chunk_id': c['chunk_id'],
            'source_question': q_raw,
            'generated_query': generated
        })

    print(f"Total C chunks: {len(c_chunks)}")
    print(f"Discarded: {len(discarded)}")
    print(f"Generated: {len(results)}")
