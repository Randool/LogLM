"""
Author: Da Ma
Date: 2022.01.04
"""
from transformers import BertTokenizer, BertTokenizerFast


# def convert_to_unicode(text):
#     """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
#     if isinstance(text, str):
#         return text
#     elif isinstance(text, bytes):
#         return text.decode("utf-8", "ignore")
#     else:
#         raise ValueError("Unsupported string type: %s" % (type(text)))


class ChPTMTokenizer(BertTokenizer):
    """
    NOTE: we use the identical bert tokenizer now!
    """

    pass


class ChPTMTokenizerFast(BertTokenizerFast):
    """
    NOTE: we use the identical bert tokenizer now!
    """

    pass
