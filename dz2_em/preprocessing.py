from dataclasses import dataclass
import re
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    def parse_alignment(text: str):
        text = (text or "").strip()
        if not text:
            return []
        pairs = []
        for item in text.split():
            i, j = item.split("-")
            pairs.append((int(i), int(j)))
        return pairs

    with open(filename, "r", encoding="utf-8") as f:
        xml_text = f.read()

    xml_text = re.sub(r"&(?!(amp|lt|gt|apos|quot);)", "&amp;", xml_text)

    root = ET.fromstring(xml_text)

    sentence_pairs = []
    alignments = []

    for sent in root.findall("s"):
        english_text = (sent.findtext("english") or "").strip()
        czech_text = (sent.findtext("czech") or "").strip()
        sure_text = sent.findtext("sure")
        possible_text = sent.findtext("possible")

        sentence_pairs.append(
            SentencePair(
                source=english_text.split(),
                target=czech_text.split(),
            )
        )

        alignments.append(
            LabeledAlignment(
                sure=parse_alignment(sure_text),
                possible=parse_alignment(possible_text),
            )
        )

    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    dict_engl, dict_czech = {}, {}
    for sent in sentence_pairs:
        engl, czech = sent.source, sent.target
        for word in engl:
            dict_engl[word] = dict_engl.get(word, 0) + 1
        for word in czech:
            dict_czech[word] = dict_czech.get(word, 0) + 1

    sorted_engl = sorted(dict_engl.items(), key=lambda x: (-x[1], x[0]))
    sorted_czech = sorted(dict_czech.items(), key=lambda x: (-x[1], x[0]))
    if freq_cutoff is not None:
        sorted_engl = sorted_engl[:freq_cutoff]
        sorted_czech = sorted_czech[:freq_cutoff]
    source_dict = {word: idx for idx, (word, _) in enumerate(sorted_engl)}
    target_dict = {word: idx for idx, (word, _) in enumerate(sorted_czech)}
    return source_dict, target_dict



def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for sent in sentence_pairs:
        source_tokens = [source_dict[word] for word in sent.source if word in source_dict]
        target_tokens = [target_dict[word] for word in sent.target if word in target_dict]

        if len(source_tokens) == 0 or len(target_tokens) == 0:
            continue

        tokenized_sentence_pairs.append(
            TokenizedSentencePair(
                source_tokens=np.array(source_tokens, dtype=np.int32),
                target_tokens=np.array(target_tokens, dtype=np.int32),
            )
        )

    return tokenized_sentence_pairs
