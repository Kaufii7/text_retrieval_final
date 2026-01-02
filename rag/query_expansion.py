"""Conservative semantic query expansion utilities.

Design goals (ROBUST04 / title queries):
- Be conservative to avoid query drift.
- Expand only noun terms (either via a curated synonym map or via NLTK+WordNet).
- Add only a small number of extra terms per query (configurable).

This module is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from rag.config import ApproachConfig


# A small, curated noun-only synonym map.
# You can extend/override this from config (cfg.params["semantic_expansion"]["synonyms"]).
DEFAULT_NOUN_SYNONYMS: Dict[str, List[str]] = {
    # Transportation / vehicles
    "aircraft": ["plane", "airplane", "aviation"],
    "automobile": ["car", "vehicle"],
    "vehicles": ["cars", "automobiles"],
    "ship": ["vessel"],
    "ships": ["vessels"],
    # Health / medicine
    "cancer": ["tumor"],
    "rabies": ["hydrophobia"],
    # Crime / law
    "counterfeiting": ["forgery"],
    "piracy": ["copyright", "infringement"],
    # Computing
    "e-mail": ["email"],
    "email": ["e-mail"],
}


def _cfg_semantic(cfg: ApproachConfig | None) -> Mapping[str, object]:
    if cfg is None:
        return {}
    params = cfg.params or {}
    sem = params.get("semantic_expansion") or {}
    return sem if isinstance(sem, Mapping) else {}


def _normalize_token(t: str) -> str:
    return (t or "").strip().lower()


def _iter_unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _merged_synonyms(cfg: ApproachConfig | None) -> Dict[str, List[str]]:
    syn: Dict[str, List[str]] = {k: list(v) for k, v in DEFAULT_NOUN_SYNONYMS.items()}
    sem = _cfg_semantic(cfg)
    extra = sem.get("synonyms", {})
    if isinstance(extra, Mapping):
        for k, v in extra.items():
            kk = _normalize_token(str(k))
            if not kk:
                continue
            if isinstance(v, (list, tuple)):
                syn[kk] = [_normalize_token(str(x)) for x in v if _normalize_token(str(x))]
            elif isinstance(v, str):
                # Allow a single synonym string.
                vv = _normalize_token(v)
                if vv:
                    syn[kk] = [vv]
    return syn


def _expand_with_nltk_wordnet(
    *,
    query_terms: Sequence[str],
    max_terms: int,
    lemma_max_per_noun: int,
) -> List[str]:
    """NLTK+WordNet noun-only expansion.

    Notes:
    - Uses POS tagging to identify nouns.
    - Pulls synonyms from WordNet lemma names (noun synsets only).
    - Keeps only single-token expansions (skips multi-word lemmas like 'trade_union').
    """
    if max_terms <= 0:
        return list(query_terms)

    # Lazy imports so the repo doesn't require nltk unless enabled.
    try:
        import nltk  # type: ignore
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "NLTK semantic expansion is enabled, but nltk/wordnet is not available.\n"
            "Install nltk and download required data, e.g.:\n"
            "  python -m pip install nltk\n"
            "  python -m nltk.downloader wordnet omw-1.4 punkt averaged_perceptron_tagger\n"
            "If you're on newer NLTK, you may also need:\n"
            "  python -m nltk.downloader averaged_perceptron_tagger_eng\n"
        ) from e

    base = [_normalize_token(t) for t in query_terms if _normalize_token(t)]
    base_unique = _iter_unique(base)
    present = set(base_unique)
    if not base_unique:
        return []

    # POS tag to identify nouns.
    try:
        tagged = nltk.pos_tag(base_unique)
    except LookupError as e:  # pragma: no cover
        raise LookupError(
            "NLTK semantic expansion is enabled, but POS tagger data is missing.\n"
            "Run:\n"
            "  python -m nltk.downloader averaged_perceptron_tagger\n"
            "If that still fails on your NLTK version, also run:\n"
            "  python -m nltk.downloader averaged_perceptron_tagger_eng\n"
        ) from e

    noun_terms = [t for t, tag in tagged if tag in ("NN", "NNS", "NNP", "NNPS")]
    if not noun_terms:
        return base_unique

    added: List[str] = []
    for t in noun_terms:
        # Pull noun synsets only.
        try:
            synsets = wn.synsets(t, pos=wn.NOUN)
        except LookupError as e:  # pragma: no cover
            raise LookupError(
                "NLTK semantic expansion is enabled, but WordNet data is missing.\n"
                "Run:\n"
                "  python -m nltk.downloader wordnet omw-1.4\n"
            ) from e

        # Collect candidate lemma names; keep it conservative.
        cands: List[str] = []
        for s in synsets[:3]:  # cap synsets to limit drift
            for lemma in s.lemmas()[: lemma_max_per_noun]:
                name = _normalize_token(lemma.name())
                if not name:
                    continue
                # Skip multiword/underscore lemmas to keep Lucene queries simple.
                if "_" in name or " " in name:
                    continue
                if name == t or name in present:
                    continue
                cands.append(name)

        # Add a couple from this noun, but respect global budget.
        for name in _iter_unique(cands)[: lemma_max_per_noun]:
            if name in present:
                continue
            present.add(name)
            added.append(name)
            if len(added) >= max_terms:
                return base_unique + added

    return base_unique + added


def expand_query_terms_semantic(
    *,
    query_terms: Sequence[str],
    cfg: ApproachConfig | None,
) -> List[str]:
    """Return query terms with conservative semantic expansions appended.

    Expansion rules:
    - Only expand noun terms (either via curated map or via NLTK+WordNet).
    - Add at most `max_terms` extra terms per query (default: 0 => disabled).
    - Never add duplicates or terms already present.
    """
    sem = _cfg_semantic(cfg)
    enabled = bool(sem.get("enabled", False))
    if not enabled:
        return list(query_terms)

    max_terms = int(sem.get("max_terms", 2))
    if max_terms <= 0:
        return list(query_terms)

    backend = str(sem.get("backend", "static")).strip().lower()
    if backend in ("nltk", "nltk_wordnet", "wordnet"):
        lemma_max_per_noun = int(sem.get("nltk_lemma_max_per_noun", 3))
        if lemma_max_per_noun <= 0:
            lemma_max_per_noun = 1
        return _expand_with_nltk_wordnet(
            query_terms=query_terms,
            max_terms=max_terms,
            lemma_max_per_noun=lemma_max_per_noun,
        )

    if backend not in ("static", ""):
        raise ValueError(f"Unknown semantic_expansion backend: {backend!r} (use 'static' or 'nltk_wordnet')")

    syn = _merged_synonyms(cfg)
    base = [_normalize_token(t) for t in query_terms if _normalize_token(t)]
    base_unique = _iter_unique(base)
    present = set(base_unique)

    added: List[str] = []
    for t in base_unique:
        cand = syn.get(t)
        if not cand:
            continue
        for s in cand:
            if not s or s in present:
                continue
            present.add(s)
            added.append(s)
            if len(added) >= max_terms:
                break
        if len(added) >= max_terms:
            break

    return base_unique + added


def expand_query_text_semantic(
    *,
    query_text: str,
    cfg: ApproachConfig | None,
) -> str:
    """Expand a raw query string by appending a few expansion terms.

    This is meant for Lucene query strings (BM25/RM3). It keeps the original query
    intact and simply appends expansions as extra tokens.
    """
    # Keep tokenization simple (split on whitespace) because Lucene will parse anyway.
    terms = [t for t in (query_text or "").split() if t.strip()]
    expanded = expand_query_terms_semantic(query_terms=terms, cfg=cfg)
    return " ".join(expanded)


