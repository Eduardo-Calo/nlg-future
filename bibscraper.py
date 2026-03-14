import re
import pandas as pd
from pybtex.database import parse_file, BibliographyData
from typing import List, Tuple, Optional, Union


def analyze_hybrid_cooccurrence(
    bib_data: 'BibliographyData',
    foundational_topics: List[Union[str, List[str]]],
    co_occurring_topics: List[Union[str, List[str]]],
    year_ranges: Union[Tuple[int, int], List[Tuple[int, int]]],
    venues_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyzes topic co-occurrence with a hybrid text-or-venue condition.

    This function defines a sub-field where for EACH `foundational_topic`, the
    term must be present in the (title/abstract) OR in the (booktitle/URL).
    It then calculates the proportion of these papers that also mention a
    `co_occurring_topic` in their title/abstract.

    This is ideal for analyzing topics that are also conference names (e.g.,
    finding papers on "summarization" within the broader "generation" field,
    where "generation" can be in the text or be part of the INLG venue name).

    Args:
        bib_data (BibliographyData): The parsed BibTeX data.
        foundational_topics (list): Topics that must be present via the hybrid
                                    text-or-venue check.
        co_occurring_topics (list): Topics to check for co-occurrence within the
                                    papers identified above (text only).
        year_ranges (tuple or list of tuples): The time period(s) to analyze.
        venues_filter (list or None, optional): An initial, hard filter to restrict
                                                the analysis to specific venues.

    Returns:
        pd.DataFrame: A DataFrame showing the co-occurrence proportion for each
                      topic within each specified time period.
    """
    # --- 1. Normalize Inputs and Compile Regex ---
    def compile_topic_regex(topic):
        if not topic: return None
        pattern_str = '|'.join(re.escape(p) for p in topic) if isinstance(topic, list) else re.escape(topic)
        return re.compile(r'\b({})\b'.format(pattern_str), re.IGNORECASE) if pattern_str else None

    ranges = [year_ranges] if isinstance(year_ranges, tuple) else year_ranges
    if not foundational_topics or not ranges: return pd.DataFrame(), [0] * len(ranges) if ranges else [0], [], {}

    foundational_patterns = [p for p in [compile_topic_regex(t) for t in foundational_topics] if p]
    if not foundational_patterns: return pd.DataFrame(), [0] * len(ranges), [], {}

    co_topic_patterns, canonical_names = {}, []
    for item in co_occurring_topics:
        name, pattern = (item[0], compile_topic_regex(item)) if isinstance(item, list) else (item, compile_topic_regex(item))
        if name not in co_topic_patterns and pattern:
            canonical_names.append(name)
            co_topic_patterns[name] = pattern
    
    venue_filter_patterns = [compile_topic_regex(v) for v in venues_filter] if venues_filter else []

    # --- 2. Single-Pass Filtering and Counting ---
    num_ranges = len(ranges)
    co_occurrence_counts = {name: [0] * num_ranges for name in canonical_names}
    foundation_total_counts = [0] * num_ranges
    filtered_entries = []  # Track entries that match foundational topics

    for entry in bib_data.entries.values():
        fields = entry.fields
        booktitle, url = fields.get("booktitle", ""), fields.get("url", "")
        
        if venue_filter_patterns and not any(p.search(booktitle + url) for p in venue_filter_patterns): continue
        try: year = int(fields.get("year", ""))
        except (ValueError, TypeError): continue
        
        year_range_index = next((i for i, (start, end) in enumerate(ranges) if start <= year <= end), -1)
        if year_range_index == -1: continue

        text_to_search = f"{fields.get('title', '')} {fields.get('abstract', '')}"
        venue_text_to_search = f"{booktitle} {url}"
        
        is_in_foundation = all(p.search(text_to_search) or p.search(venue_text_to_search) for p in foundational_patterns)
        
        if is_in_foundation:
            foundation_total_counts[year_range_index] += 1
            filtered_entries.append((entry, fields, year))
            for name, pattern in co_topic_patterns.items():
                if pattern.search(text_to_search):
                    co_occurrence_counts[name][year_range_index] += 1

    # --- 3. Calculate Proportions ---
    proportions = {name: [0.0] * num_ranges for name in canonical_names}
    for name, counts in co_occurrence_counts.items():
        for i in range(num_ranges):
            if foundation_total_counts[i] > 0:
                proportions[name][i] = counts[i] / foundation_total_counts[i]

    # --- 4. DataFrame Creation ---
    column_names = [f"{start}-{end}" for start, end in ranges]
    df = pd.DataFrame.from_dict(proportions, orient='index', columns=column_names)
    df.index.name = "co_occurring_topic"

    return df.reindex(canonical_names).sort_values(by=column_names[0], ascending=False), foundation_total_counts, filtered_entries, co_occurrence_counts


# Helper function to extract venue from ACL Anthology URL
def extract_venue_from_url(url: str) -> Optional[str]:
    """
    Extract venue from ACL Anthology URL.
    Handles both old and new formats:
    - New format: https://aclanthology.org/2025.acl-long.1/ -> 'acl'
    - Old format (pre-2020): https://aclanthology.org/P18-1024/ -> 'acl'
    """
    # Mapping for old ACL Anthology letter prefixes
    # https://aclanthology.org/info/ids/
    # https://aclanthology.org/faq/linking/
    old_venue_mapping = {
        'A': 'anlp',
        'C': 'coling',
        'D': 'emnlp',
        'E': 'eacl',
        'H': 'hlt',
        'I': 'ijcnlp',
        'L': 'lrec',
        'N': 'naacl',
        'P': 'acl',
        'W': 'ws'  # Workshop prefix
    }
    
    if not url or "aclanthology.org" not in url:
        return None
    try:
        path = url.split("aclanthology.org/")[-1].strip("/")
        
        # New format: 2025.acl-long.1
        if path[0].isdigit():  # Starts with year
            parts = path.split(".")
            if len(parts) >= 2:
                venue_part = parts[1].split("-")[0]
                return venue_part.lower()
        # Old format: P18-1024 or W19-1234
        else:
            letter_prefix = path[0].upper()
            return old_venue_mapping.get(letter_prefix, letter_prefix.lower())
    except Exception:
        pass
    return None


def print_venue_statistics(filtered_entries, domain_name, year_ranges=None, top_n=100):
    """
    Print venue statistics for filtered entries.
    
    Args:
        filtered_entries: List of (entry, fields, year) tuples
        domain_name: Name of the domain for the header
        year_ranges: Optional list of year ranges to group by period
        top_n: Number of top venues to display (default: 10)
    """
    print(f"\n--- Statistics for {domain_name} ---")
    
    if year_ranges and len(year_ranges) > 1:
        # Multi-period analysis
        venues_by_period = [dict() for _ in year_ranges]
        unmatched_by_period = [0] * len(year_ranges)
        
        for entry, fields, year in filtered_entries:
            url = fields.get('url', '')
            venue = extract_venue_from_url(url)
            period_index = next((i for i, (start, end) in enumerate(year_ranges) if start <= year <= end), -1)
            if period_index == -1:
                continue
            if not venue:
                unmatched_by_period[period_index] += 1
                continue
            venues_by_period[period_index][venue] = venues_by_period[period_index].get(venue, 0) + 1
        
        for period_idx, ((start, end), venue_counts) in enumerate(zip(year_ranges, venues_by_period)):
            papers_with_venue = sum(venue_counts.values())
            papers_without_venue = unmatched_by_period[period_idx]
            print(f"\nPeriod {start}-{end}")
            print(f"Number of unique venues: {len(venue_counts)}")
            print(f"Papers with extractable venue: {papers_with_venue}")
            print(f"Papers without valid ACL Anthology URL: {papers_without_venue}")
            print("Top venues:")
            for venue, count in sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]:
                print(f"  {venue}: {count}")
    else:
        # Single period analysis
        venues = {}
        for entry, fields, year in filtered_entries:
            url = fields.get('url', '')
            venue = extract_venue_from_url(url)
            if venue:
                venues[venue] = venues.get(venue, 0) + 1
        print(f"Number of unique venues: {len(venues)}")
        print("Top venues:")
        for venue, count in sorted(venues.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"  {venue}: {count}")


def print_sample_papers(filtered_entries, topics_to_analyze, domain_name, num_samples=5):
    """
    Print sample papers from a filtered set with their matched topics.
    
    Args:
        filtered_entries: List of (entry, fields, year) tuples
        topics_to_analyze: List of topics to check for matches
        domain_name: Name of the domain for the header
        num_samples: Number of samples to print (default: 5)
    """
    print(f"--- Sample Papers from {domain_name} ---")
    
    for i, (entry, fields, year) in enumerate(filtered_entries[:num_samples]):
        print(f"\n{i+1}. {fields.get('title', 'N/A')} ({year})")
        print(f"   Venue: {extract_venue_from_url(fields.get('url', ''))}")
        
        # Show which topics this paper matches
        matched_topics = []
        text_to_search = f"{fields.get('title', '')} {fields.get('abstract', '')}"
        for topic in topics_to_analyze:
            if isinstance(topic, list):
                pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in topic) + r')\b', re.IGNORECASE)
                if pattern.search(text_to_search):
                    matched_topics.append(topic[0])
            else:
                pattern = re.compile(r'\b' + re.escape(topic) + r'\b', re.IGNORECASE)
                if pattern.search(text_to_search):
                    matched_topics.append(topic)
        
        print(f"   Matched topics: {', '.join(matched_topics) if matched_topics else 'None'}")
        print(f"   URL: {fields.get('url', 'N/A')[:80]}...")


def count_topic_matches(filtered_entries, topics_to_analyze, year_ranges=None):
    """
    Count how many papers have topic matches vs none, broken down by time period.
    
    Args:
        filtered_entries: List of (entry, fields, year) tuples
        topics_to_analyze: List of topics to check for matches
        year_ranges: Optional tuple or list of (start, end) tuples. If None,
                     all entries are treated as a single period.
        
    Returns:
        Tuple of (papers_with_matches, papers_without_matches, total_matches)
        where each element is a list of counts aligned with year_ranges.
    """
    if year_ranges is None:
        ranges = None
    elif isinstance(year_ranges, tuple) and len(year_ranges) == 2 and isinstance(year_ranges[0], int):
        ranges = [year_ranges]
    else:
        ranges = list(year_ranges)

    num_ranges = len(ranges) if ranges else 1
    papers_with_matches = [0] * num_ranges
    papers_without_matches = [0] * num_ranges
    total_matches = [0] * num_ranges
    
    for entry, fields, year in filtered_entries:
        if ranges:
            period_idx = next((i for i, (start, end) in enumerate(ranges) if start <= year <= end), -1)
            if period_idx == -1:
                continue
        else:
            period_idx = 0

        matched_topics = []
        text_to_search = f"{fields.get('title', '')} {fields.get('abstract', '')}"
        
        for topic in topics_to_analyze:
            if isinstance(topic, list):
                pattern = re.compile(r'\b(' + '|'.join(re.escape(p) for p in topic) + r')\b', re.IGNORECASE)
                if pattern.search(text_to_search):
                    matched_topics.append(topic[0])
            else:
                pattern = re.compile(r'\b' + re.escape(topic) + r'\b', re.IGNORECASE)
                if pattern.search(text_to_search):
                    matched_topics.append(topic)
        
        if matched_topics:
            papers_with_matches[period_idx] += 1
            total_matches[period_idx] += len(matched_topics)
        else:
            papers_without_matches[period_idx] += 1
    
    return papers_with_matches, papers_without_matches, total_matches


def extract_acl_id_from_url(url: str) -> Optional[str]:
    """
    Extract ACL ID from ACL Anthology URL.
    Example:
    - https://aclanthology.org/O02-2002/ -> 'O02-2002'
    """
    if not url or "aclanthology.org" not in url:
        return None
    try:
        # Extract the path after aclanthology.org/
        path = url.split("aclanthology.org/")[-1].strip("/")
        
        # Old format: O02-2002, L02-1310, etc.
        if re.match(r'^[A-Z]\d{2}-\d{4}$', path):
            return path

    except Exception:
        pass
    return None


def enrich_abstracts_from_parquet(bib_data: 'BibliographyData', parquet_path: str) -> Tuple[int, int]:
    """
    Load abstracts from a parquet file and cross-reference with bibliography entries.
    Matches papers by ACL ID extracted from URLs.
    
    Args:
        bib_data: The parsed BibTeX data to enrich
        parquet_path: Path to the parquet file with abstract data
    
    Returns:
        Tuple of (abstracts_added, papers_matched)
    """
    try:
        df_parquet = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return 0, 0
    
    # Create a mapping from ACL ID to abstract
    acl_id_to_abstract = {}
    for _, row in df_parquet.iterrows():
        acl_id = row.get('acl_id', '')
        abstract = row.get('abstract', '')
        if acl_id and abstract and str(abstract).strip():
            acl_id_to_abstract[acl_id] = str(abstract)
    
    abstracts_added = 0
    papers_matched = 0
    
    # Match and enrich by ACL ID
    for entry in bib_data.entries.values():
        fields = entry.fields
        
        # Skip if abstract already exists
        if fields.get('abstract', '').strip():
            continue
        
        # Extract ACL ID from URL
        url = fields.get('url', '')
        acl_id = extract_acl_id_from_url(url)
        
        if acl_id and acl_id in acl_id_to_abstract:
            fields['abstract'] = acl_id_to_abstract[acl_id]
            abstracts_added += 1
            papers_matched += 1
    
    return abstracts_added, papers_matched


def count_abstracts_by_period(bib_data: 'BibliographyData', year_ranges: Union[Tuple[int, int], List[Tuple[int, int]]]):
    """
    Count how many papers have (or lack) an abstract within each year range.
    Returns two lists aligned with the provided ranges: (with_abstract, without_abstract).
    """
    ranges = [year_ranges] if isinstance(year_ranges, tuple) else year_ranges
    if not ranges:
        return [], []

    with_abstract = [0] * len(ranges)
    without_abstract = [0] * len(ranges)

    for entry in bib_data.entries.values():
        fields = entry.fields
        try:
            year = int(fields.get("year", ""))
        except (ValueError, TypeError):
            continue

        idx = next((i for i, (start, end) in enumerate(ranges) if start <= year <= end), -1)
        if idx == -1:
            continue

        abstract_text = fields.get("abstract", "")
        if abstract_text and abstract_text.strip():
            with_abstract[idx] += 1
        else:
            without_abstract[idx] += 1

    return with_abstract, without_abstract


# Parse the bibliography data

print("Parsing bibliography data...")
bib_data = parse_file("anthology.bib")
print("Number of entries parsed:", len(bib_data.entries))

# Enrich abstracts from parquet file
print("\nEnriching abstracts from parquet file...")
abstracts_before = sum(1 for entry in bib_data.entries.values() if entry.fields.get('abstract', '').strip())
added, matched = enrich_abstracts_from_parquet(bib_data, "acl-publication-info.74k.v2.parquet")
abstracts_after = sum(1 for entry in bib_data.entries.values() if entry.fields.get('abstract', '').strip())
print(f"Abstracts before enrichment: {abstracts_before}")
print(f"Abstracts added: {added}")
print(f"Abstracts after enrichment: {abstracts_after}")
print(f"Papers matched from parquet: {matched}")

# --- Example Usage ---

# Define the topics and other parameters for the analysis
anchors = [
    ["large language model", "LLM", "LLMs"],
    ["language generation", "nlg", "inlg"]
]

topics_to_analyze = [
    ["aggregation", "splitting", "merging"],
    "analogy",
    "argument mining",
    "authorship verification",
    "automated essay scoring",
    "automatic evaluation",
    ["automatic speech recognition", "ASR", "speech-to-text", "speech recognition", "speech segmentation"],
    "bias detection",
    "lexicon induction",
    "code generation",
    "commonsense reasoning",
    "content determination",
    ["coreference resolution", "anaphora resolution", "pronominal resolution"],
    "data augmentation",
    "data-to-text",
    "dialogue state tracking",
    "dialogue understanding",
    ["discourse parsing", "discourse analysis"],
    "discourse planning",
    ["entity linking", "entity disambiguation", "named entity linking"],
    "event extraction",
    ["fake news detection", "misinformation detection", "rumor detection"],
    "figurative language",
    "grammar induction",
    ["grammatical error correction", "GEC"],
    ["hallucination detection", "hallucination"],
    ["hate speech detection", "abusive language detection"],
    "human evaluation",
    "image captioning",
    ["information extraction", "IE"],
    ["information retrieval", "IR"],
    "intent detection",
    ["knowledge base question answering", "KBQA"],
    ["language change", "diachronic NLP", "historical NLP"],
    ["language identification", "language detection"],
    ["lemmatization", "lemmatisation"],
    ["lexicalization", "lexicalisation"],
    ["long form question answering", "generative QA"],
    "lyrics generation",
    ["machine translation", "MT"],
    "mathematical question answering",
    "mathematical reasoning",
    "multiple choice question answering",
    ["named entity recognition", "NER", "named entities", "entity extraction"],
    ["natural language inference", "NLI", "textual entailment", "RTE"],
    ["optical character recognition", "OCR"],
    ["paraphrasing", "paraphrase"],
    ["part-of-speech tagging", "POS tagging", "POS-tagging", "part of speech tagging", "part-of-speech-tagging"],
    "plagiarism detection",
    "poetry generation",
    "prompt engineering",
    "question generation",
    "recommender systems",
    "referring expression generation",
    "relation extraction",
    ["semantic parsing", "meaning representation parsing"],
    ["semantic role labeling", "SRL"],
    "sentence segmentation",
    ["sentiment analysis", "opinion mining", "polarity classification"],
    "sign language recognition",
    "spam detection",
    ["speech synthesis", "text-to-speech", "TTS"],
    "stemming",
    "story generation",
    "style transfer",
    ["surface realization", "surface realisation", "linguistic realisation", "linguistic realization"],
    ["syntactic parsing", "dependency parsing", "constituency parsing"],
    ["taxonomy construction", "ontology induction"],
    "terminology extraction",
    ["text classification", "document classification"],
    "text clustering",
    "text simplification",
    ["text summarization", "text summarisation"],
    "tokenization",
    ["topic modeling", "LDA"],
    ["toxicity understanding", "offensive language detection"],
    "video captioning",
    ["visual question answering", "VQA"],
    ["word sense disambiguation", "WSD"],
    ["word sense induction", "WSI"],
]

comparison_year_ranges = [(1965, 2020), (2021, 2025)]

# Abstract coverage overview for the defined periods
with_abs, without_abs = count_abstracts_by_period(bib_data, comparison_year_ranges)
print("\nAbstract coverage by period:")
for (start, end), has_abs, no_abs in zip(comparison_year_ranges, with_abs, without_abs):
    total = has_abs + no_abs
    coverage_pct = (100 * has_abs / total) if total else 0
    print(f"  {start}-{end}: {has_abs} with abstracts, {no_abs} without ({coverage_pct:.2f}% coverage)")
print()

# Call the function for the "Generation" only analysis
gen_df, gen_totals, gen_filtered, gen_counts = analyze_hybrid_cooccurrence(
    bib_data=bib_data,
    foundational_topics=[anchors[1]],
    co_occurring_topics=topics_to_analyze,
    year_ranges=comparison_year_ranges,
)

print(f"Total papers in 'Generation' domain (1965-2020): {gen_totals[0]}")
print(f"Total papers in 'Generation' domain (2021-2025): {gen_totals[1]}")
print("\nFocus of Research within the Generation Domain (counts / % of papers)")
# Create display dataframe with counts and percentages
gen_display = pd.DataFrame()
for topic in gen_df.index:
    for i, col in enumerate(gen_df.columns):
        count = gen_counts[topic][i]
        pct = gen_df.loc[topic, col] * 100
        gen_display.loc[topic, col] = f"{count} / {pct:.2f}%"
print(gen_display.to_string())

# Count topic matches
with_matches, without_matches, total = count_topic_matches(gen_filtered, topics_to_analyze, comparison_year_ranges)
print(f"\nTopic Match Statistics:")
for i, (start, end) in enumerate(comparison_year_ranges):
    period_total = with_matches[i] + without_matches[i]
    if period_total > 0:
        print(f"  Period {start}-{end}:")
        print(f"    Papers with at least one topic match: {with_matches[i]} ({100*with_matches[i]/period_total:.2f}%)")
        print(f"    Papers with no topic matches: {without_matches[i]} ({100*without_matches[i]/period_total:.2f}%)")
        print(f"    Total topic matches across all papers: {total[i]}")
        print(f"    Average matches per paper (for papers with matches): {total[i]/with_matches[i]:.2f}" if with_matches[i] > 0 else "")

print_venue_statistics(gen_filtered, "Generation Domain", comparison_year_ranges)

print("\n" + "="*80 + "\n")

# Call the function for the "Generation + LLM" analysis
gen_llm_df, gen_llm_totals, gen_llm_filtered, gen_llm_counts = analyze_hybrid_cooccurrence(
    bib_data=bib_data,
    foundational_topics=anchors,
    co_occurring_topics=topics_to_analyze,
    year_ranges=comparison_year_ranges[1],
)

print(f"Total papers in 'Generation + LLM' domain (2021-2025): {gen_llm_totals[0]}")
print("\nFocus of Research within the Generation + LLM Domain (counts / % of papers)")
# Create display dataframe with counts and percentages
gen_llm_display = pd.DataFrame()
for topic in gen_llm_df.index:
    for i, col in enumerate(gen_llm_df.columns):
        count = gen_llm_counts[topic][i]
        pct = gen_llm_df.loc[topic, col] * 100
        gen_llm_display.loc[topic, col] = f"{count} / {pct:.2f}%"
print(gen_llm_display.to_string())

# Count topic matches
with_matches, without_matches, total = count_topic_matches(gen_llm_filtered, topics_to_analyze, comparison_year_ranges[1])
print(f"\nTopic Match Statistics:")
period_total = with_matches[0] + without_matches[0]
if period_total > 0:
    print(f"  Papers with at least one topic match: {with_matches[0]} ({100*with_matches[0]/period_total:.2f}%)")
    print(f"  Papers with no topic matches: {without_matches[0]} ({100*without_matches[0]/period_total:.2f}%)")
    print(f"  Total topic matches across all papers: {total[0]}")
    print(f"  Average matches per paper (for papers with matches): {total[0]/with_matches[0]:.2f}" if with_matches[0] > 0 else "")

print_venue_statistics(gen_llm_filtered, "Generation + LLM Domain")

print("\n" + "="*80 + "\n")

# Call the function for the "LLM" only analysis
llm_df, llm_totals, llm_filtered, llm_counts = analyze_hybrid_cooccurrence(
    bib_data=bib_data,
    foundational_topics=[anchors[0]],
    co_occurring_topics=topics_to_analyze,
    year_ranges=comparison_year_ranges[1],
)

print(f"Total papers in 'LLM' domain (2021-2025): {llm_totals[0]}")
print("\nFocus of Research within the LLM Domain (counts / % of papers)")
# Create display dataframe with counts and percentages
llm_display = pd.DataFrame()
for topic in llm_df.index:
    for i, col in enumerate(llm_df.columns):
        count = llm_counts[topic][i]
        pct = llm_df.loc[topic, col] * 100
        llm_display.loc[topic, col] = f"{count} / {pct:.2f}%"
print(llm_display.to_string())

# Count topic matches
with_matches, without_matches, total = count_topic_matches(llm_filtered, topics_to_analyze, comparison_year_ranges[1])
print(f"\nTopic Match Statistics:")
period_total = with_matches[0] + without_matches[0]
if period_total > 0:
    print(f"  Papers with at least one topic match: {with_matches[0]} ({100*with_matches[0]/period_total:.2f}%)")
    print(f"  Papers with no topic matches: {without_matches[0]} ({100*without_matches[0]/period_total:.2f}%)")
    print(f"  Total topic matches across all papers: {total[0]}")
    print(f"  Average matches per paper (for papers with matches): {total[0]/with_matches[0]:.2f}" if with_matches[0] > 0 else "")

print_venue_statistics(llm_filtered, "LLM Domain")

print("\n" + "="*80 + "\n")

# Sanity check: print sample papers from each domain
print_sample_papers(gen_filtered, topics_to_analyze, "Generation Domain")

print("\n" + "="*80 + "\n")

print_sample_papers(gen_llm_filtered, topics_to_analyze, "Generation + LLM Domain")

print("\n" + "="*80 + "\n")

print_sample_papers(llm_filtered, topics_to_analyze, "LLM Domain")

# --- Grouped horizontal bar chart: Gen 1965-2020 / Gen 2021-2025 / Gen+LLM 2021-2025 ---
import plotly.graph_objects as go

bar_data = []
all_topics = sorted(set(gen_df.index) | set(gen_llm_df.index))
for topic in all_topics:
    gen_early = gen_df.loc[topic, "1965-2020"] * 100 if topic in gen_df.index else 0.0
    gen_late  = gen_df.loc[topic, "2021-2025"] * 100 if topic in gen_df.index else 0.0
    llm_late  = gen_llm_df.loc[topic, "2021-2025"] * 100 if topic in gen_llm_df.index else 0.0
    if max(gen_early, gen_late, llm_late) >= 1.0:
        bar_data.append((topic, gen_early, gen_late, llm_late))

# Sort by Gen 1965-2020 ascending so highest values appear at top
bar_data.sort(key=lambda x: x[1])

topics_bar     = [t for t, _, _, _ in bar_data]
gen_early_vals = [v for _, v, _, _ in bar_data]
gen_late_vals  = [v for _, _, v, _ in bar_data]
llm_late_vals  = [v for _, _, _, v in bar_data]

# Determine text colors: green if topic increased (early→late), red if decreased
text_colors = ["green" if late >= early else "red" for early, late in zip(gen_early_vals, gen_late_vals)]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    y=topics_bar, x=llm_late_vals, orientation="h",
    name="NLG+LLM (2021-2025)",
    marker_color="coral",
    text=[f"{v:.1f}%" for v in llm_late_vals],
    textposition="outside",
))
fig_bar.add_trace(go.Bar(
    y=topics_bar, x=gen_late_vals, orientation="h",
    name="NLG (2021-2025)",
    marker_color="steelblue",
    text=[f"{v:.1f}%" for v in gen_late_vals],
    textposition="outside",
    textfont=dict(color=text_colors),
))
fig_bar.add_trace(go.Bar(
    y=topics_bar, x=gen_early_vals, orientation="h",
    name="NLG (1965-2020)",
    marker_color="lightsteelblue",
    text=[f"{v:.1f}%" for v in gen_early_vals],
    textposition="outside",
))

fig_bar.update_layout(
    xaxis_title="% of papers",
    barmode="group",
    bargroupgap=0.08,
    bargap=0.17,
    template="plotly_white",
    font=dict(size=21),
    uniformtext_minsize=19,
    uniformtext_mode="show",
    width=1900,
    height=1300,
    legend=dict(x=0.55, y=0.02, traceorder="reversed"),
)
fig_bar.write_html("topic_bar_chart.html")
fig_bar.write_image("topic_bar_chart.pdf")
fig_bar.show()
print("Bar chart saved to topic_bar_chart.html and topic_bar_chart.pdf")
