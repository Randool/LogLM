"""
Author: Da Ma
Date: 2021.12.31
"""
import math
import re
from string import punctuation as enPunctuation
from typing import List

import numpy as np
from zhon.hanzi import punctuation as chPunctuation

punctuation = enPunctuation + chPunctuation


def sentSplit(document: str) -> List[str]:
    """
    Refer to the implementation of LTP tokenizer, for the chinese text mainly.
    """
    sent_list = []
    try:
        document = re.sub(
            "(?P<quotation_mark>([。？！…?!](?![”’\"'])))",
            r"\g<quotation_mark>\n",
            document,
        )  # 单字符断句符
        document = re.sub(
            "(?P<quotation_mark>(?<![\\d])[.](?![\\d]))",
            r"\g<quotation_mark>\n",
            document,
        )
        document = re.sub(
            "(?P<quotation_mark>(([。？！!?]|…{1,2})[”’\"']))",
            r"\g<quotation_mark>\n",
            document,
        )  # 特殊引号

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                sent_list.append(sent)
    except Exception:
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def cutSentByNumOrder(line: str) -> List[str]:
    """split sentences further according to number order 1. 1、1，and so on"""
    patterns = [
        r"(?<![/－～\~\.\d-])[\d]、\s*",
        r"(?<![/－～\~\.\d-])[\d]，\s*",
        r"(?<![/－～\~\.\d-])[\d],\s*",
        r"(?<![/－～\~\.\d-])[\d]\.\D+",
        r"\([\d]\)\s*",
        r"（[\d]）\s*",
        r"\[[\d]\]\s*",
        r"(?<![/－～\~\.\d-])[\d]\)\s*",
        r"(?<![/－～\~\.\d-])[\d]）\s*",
        r"(?<![/－～\~\.\d-])[\d]\s*(?![克斤米只公个号年月日周天到段/－～\~\.\d-])",
        # exclude specific cases:
        # 1. 1~2
        # 2. 1.2
        # 3. 1-2
        # 4. 3月|公斤|个 ...
    ]

    PREFIX = "\n"
    sents = [line]
    for pattern in patterns:
        # NOTE: the case of only one number is not considered.
        allFinds = re.findall(pattern, line)
        if len(allFinds) > 1:
            order = 1
            right = True
            for fakeSent in allFinds:
                try:
                    fakeOrder = int(fakeSent[0].strip())
                    if fakeOrder != order:
                        right = False
                        break
                    else:
                        order += 1
                except Exception:
                    right = True
                    break

            if not right:
                continue

            line = re.sub(pattern, lambda m: PREFIX + m.group(0), line)
            sents = line.split(PREFIX)
            break

    return [sent for sent in sents if sent]


def cutTitle(line: str) -> List[str]:
    sents = [sent for sent in re.split(r"\s+", line, maxsplit=1) if sent]
    return sents


def removeDupTitle(sents: List[str]) -> List[str]:
    if len(sents) > 1 and sents[1].strip().lower().startswith(sents[0].strip().lower()):
        return sents[1:]
    return sents


def normalizeSent(sent: str) -> List[str]:
    """
    (1) remove leading punctuations
    (2) remove trivial sentences, e.g. 1。
    (3) split [\t] delimited sentences
    (4) remove spaces
    """
    sent = sent.lstrip(punctuation)

    if not sent:
        return []

    sentList = [subsent for subsent in sent.split("\t") if subsent]

    for index, sent in enumerate(sentList):
        sentList[index] = "".join(sent.split())

    return sentList


def permutate_sent(sents: List[List[str]], sent_perm_ratio: float) -> List[List[str]]:
    """Permutate sentences in a sequence referring to <sent_perm_ratio>
    """
    if sent_perm_ratio <= 0:
        return sents[:]
    else:
        perm_sent_num = math.ceil(len(sents) * sent_perm_ratio)
        substitutions = np.random.permutation(len(sents))[:perm_sent_num]
        ordering = np.arange(len(sents))
        ordering[substitutions] = substitutions[np.random.permutation(perm_sent_num)]

        results = sents[:]

        for idx, order in enumerate(ordering):
            results[idx] = sents[order]

        return results


def get_word_mask(sent: List[str]) -> List[int]:
    """For WordPiece Tokenizer. Return the word level mask.
    The return is a binary list. 1 indicates the start of
    a word.
    """
    mask = [1]
    for token in sent[1:]:
        if token.startswith("##"):
            mask.append(0)
        else:
            mask.append(1)

    return mask


def merge_sents(sents: List[List[str]], tokenizer):
    seq = []
    word_mask = []

    for sent in sents:
        seq += tokenizer.convert_tokens_to_ids(sent)
        word_mask += get_word_mask(sent)

    return seq, word_mask


def insert_mask(tokens: List[int], p: float, mask_random_ratio: float,
                mask_token_id: int, vocab_size: int, start: int):
    """Insert mask or random words referring to <p> for 0-len mask span.
    """
    if p <= 0:
        return tokens

    num_tokens = len(tokens)
    n = int(math.ceil(num_tokens * p))

    noise_indices = np.random.permutation(num_tokens + n)[:n]
    noise_mask = np.zeros((num_tokens + n,), dtype=np.bool)
    noise_mask[noise_indices] = 1
    result = np.full((n + num_tokens,), -1, dtype=np.int)

    num_random = int(math.ceil(n * mask_random_ratio))
    result[noise_indices[num_random:]] = mask_token_id
    result[noise_indices[:num_random]] = np.random.randint(
        start, vocab_size - 1, num_random
    )  # remove special tokens

    result[~noise_mask] = tokens

    assert (result >= 0).all()
    return result


def span_mask_seq(seq: List[int], word_mask: List[int], mask_ratio: float, tokenizer,
                  poisson_lambda: float, mask_random_ratio: float, max_len: int):
    """mask source sequence.
    params:
    -------
        sents: List[List] tokenized seq
    Return:
    -------
        masked merged sequence id: List
    """
    if mask_ratio <= 0:
        return seq
    else:
        word_mask = np.array(word_mask)
        num_to_mask = int(math.ceil(word_mask.astype(np.float).sum() * mask_ratio))

        if num_to_mask == 0:
            return seq

        merged_idx_sent = np.array(seq)

        # construct the length list of masked spans
        cur_len = 0
        lengths = []
        while cur_len < num_to_mask:
            next_len = np.random.poisson(poisson_lambda, 1).item()
            if cur_len + next_len > num_to_mask:
                next_len = num_to_mask - cur_len
            lengths.append(next_len)
            cur_len += next_len

        # Handle 0-length mask (inserts) separately
        num_to_mask = len(lengths)
        lengths = np.array(lengths)
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - len(lengths)
        num_to_mask -= num_inserts

        # find where to mask and where to insert, mask first
        word_starts = np.nonzero(word_mask)[0]
        indices = word_starts[np.random.permutation(word_starts.shape[0])][:num_to_mask]
        mask_random = np.random.uniform(size=num_to_mask) < mask_random_ratio
        to_keep = np.ones(merged_idx_sent.shape[0], dtype=np.bool)
        mask_idx = tokenizer.mask_token_id

        # mask start of span
        start = len(tokenizer.special_tokens_map)
        merged_idx_sent[indices] = mask_idx
        merged_idx_sent[indices[mask_random]] = np.random.randint(
            start, len(tokenizer) - 1, mask_random.sum()
        )

        # mask other tokens in span
        word_mask = np.append(word_mask, 0)
        word_mask[-1] = 255
        lengths -= 1
        while indices.shape[0] > 0:
            assert lengths.shape[0] == indices.shape[0]
            lengths -= word_mask[indices + 1].astype(np.int)
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            lengths = lengths[uncompleted]
            to_keep[indices] = 0

        merged_idx_sent = merged_idx_sent[to_keep]

        if num_inserts > 0:
            merged_idx_sent = insert_mask(
                merged_idx_sent,
                num_inserts / merged_idx_sent.shape[0],
                mask_random_ratio,
                tokenizer.mask_token_id,
                len(tokenizer),
                start,
            )

        # merge continuous mask id and truncate
        noised_seq = []
        prev = -1
        for token_idx in merged_idx_sent:
            if prev == mask_idx and token_idx == mask_idx:
                continue
            noised_seq.append(int(token_idx))
            prev = token_idx

        return noised_seq[:max_len]
