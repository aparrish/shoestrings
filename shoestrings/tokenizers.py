import re

class Tokenizer:
    """Base class for Tokenizers

    The Tokenizer class can be used to process incoming data (such as text)
    into a sequence of unique tokens, which can then be used to build a
    `MarkovModel`. Tokenization requires two passes over the data: one to
    ascertain the list of unique tokens and assign integer IDs to each token
    type, and another to convert the data into sequences of IDs that match
    those token types.

    The `specials` parameter, if provided, should supply: (1) a string
    representing the "beginning of string" token (BOS); (2) a string
    representing the "end of string" token (EOS); (3) an integer ID to use for
    the BOS token; and (4) an integer ID to use for the EOS token. If this
    parameter is not supplied, the values are retrieved from the internal
    `._get_default_specials()` method.

    Parameters
    ----------
    specials : list or tuple, optional
        list of values, as specified above
    reserve_special : boolean
        whether to reserve space in the tokenizer vocabulary for BOS and EOS
    """
    def __init__(self, specials=None, reserve_special=True):
        self.vocab = []
        self.vocab2idx = {}
        if specials is None:
            specials = self._get_default_specials()
        (self.BOS,
            self.BOS_ID,
            self.EOS,
            self.EOS_ID) = specials
        if reserve_special:
            self.vocab.extend([self.BOS, self.EOS])

    def _get_default_specials(self):
        return [
            "<s>", # beginning of string
            0, # beginning of string ID
            "</s>", # end of string
            1, # end of string ID
        ]

    def build(self, iterable):
        """Builds the tokenizer's vocabulary from iterable of data items

        Parameters
        ----------
        iterable : iterable
            an iterable of items to tokenize (e.g., a list of strings)
        """
        tokens = set()
        for item in iterable:
            tokens.update(self.tokenize(item))
        self.vocab.extend(sorted(list(tokens)))
        self.vocab2idx = {item: i for i, item in enumerate(self.vocab)}

    def build_from_one(self, item):
        """Build's the tokenizer's vocabulary from a single data item

        This is a convenience method for `.build()` when the data consists of
        only a single item (e.g., a single string).

        Parameters
        ----------
        item
            the item to be used as data to build the tokenizer's vocabulary
        """
        self.build([item])

    def convert_tokens_to_ids(self, tokens):
        """Converts a list of tokens to their corresponding IDs

        Parameters
        ----------
        tokens : list-like
            sequence of tokens

        Returns
        -------
        list
            list of token IDs
        """
        ids = [self.vocab2idx[item] for item in tokens]
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a list of token IDs to their corresponding tokens

        Parameters
        ----------
        ids : list-like
            sequence of token IDs

        Returns
        -------
        list
            list of tokens
        """
        ids = [self.vocab[idx] for idx in ids]
        return ids

    def encode(self, data, add_special=True):
        """Encodes an iterable of untokenized items

        This method takes an iterable of individual items and returns a list
        of lists of token IDs that correspond to those items. It calls the
        `.tokenize()` method to convert the individual data items to tokens,
        then `.convert_tokens_to_ids()` to retrieve the corresponding token
        IDs. This is helpful for converting incoming data into sequences of
        tokens for (e.g.) the `MarkovModel`'s `.build()` method.

        Parameters
        ----------
        data : iterable
            an iterable of data items (e.g., a list of strings)
        add_special : boolean
            adds the `BOS` and `EOS` tokens before and after each item

        Returns
        -------
        list
            a list of lists of token IDs
        """
        if add_special:
            return [[self.vocab2idx[self.BOS]] + \
                    self.convert_tokens_to_ids(self.tokenize(item)) + \
                    [self.vocab2idx[self.EOS]] for item in data]
        else:
            return [self.convert_tokens_to_ids(self.tokenize(item)) for item in
                    data]

    def encode_one(self, item, add_special=True):
        """Encodes a single item

        This is a convenience method for encoding only a single item, and
        retrieving a flat list of token IDs for that in response.

        Parameters
        ----------
        item
            data item to encode
        add_special : boolean
            adds the `BOS` and `EOS` tokens before and after each item

        Returns
        -------
        list
            a list of token IDs
        """
        return self.encode([item], add_special=add_special)[0]

    def __call__(self, text):
        return self.tokenize(text)

    def decode(self, data, strip_special=True):
        """Converts an iterable of lists of token IDs to decoded items

        This method takes an iterable of lists of token IDs (e.g., as returned
        from the `MarkovModel`'s `.generate()` method) and decodes them.
        "Decoding" consists of converting the token IDs to their corresponding
        tokens, and then calling the decoder's `.decode_cleanup()` method to
        convert the tokens back to a format that resembles the data that was
        originally used to decode the model. (For example, the
        `SimpleWordTokenizer` class has a `.decode_cleanup()` method that joins
        a list of token strings back into a single string.)

        Parameters
        ----------
        data : iterable
            an iterable of lists of token IDs
        strip_special : boolean, optional
            whether to remove BOS and EOS tokens before cleanup

        Returns
        -------
        list
            a list of decoded items
        """
        output = []
        for item in data:
            if strip_special:
                tokens = [x for x in self.convert_ids_to_tokens(item)
                          if x not in (self.BOS, self.EOS)]
            else:
                tokens = self.convert_ids_to_tokens(item)
            output.append(self.decode_cleanup(tokens))
        return output

    def decode_one(self, item, strip_special=True):
        """Decodes a list of token IDs

        Convenience method for calling `.decode()` with a single list of
        token IDs (rather than an iterable of lists of token IDs).

        Parameters
        ----------
        item : list-like 
            a list of token IDs
        strip_special : boolean, optional
            whether to remove BOS and EOS tokens before cleanup

        Returns
        -------
        value
            the decoded item
        """
        return self.decode([item], strip_special=strip_special)[0]

    def decode_cleanup(self, tokens):
        """Cleans up decoded tokens"""
        raise NotImplementedError

    def tokenize(self, data):
        """Tokenizes iterable of data items"""
        raise NotImplementedError

class SimpleWordTokenizer(Tokenizer):
    """Tokenizer class for tokenizing a string into words

    Well, "words" for one potential definition of "word," i.e., things that
    fall between regular expression word boundary characters.
    """

    def tokenize(self, data):
        """Tokenizes text data into individual words using a simple regex

        Parameters
        ----------
        data
            iterable of strings

        Returns
        -------
        list
            list of lists of strings
        """
        tokenized = [tok.strip() for tok in
                     re.findall(r"\b(.+?)(?:\b|$)", data)
                     if tok.strip() not in (' ', '')]
        return tokenized

    def decode_cleanup(self, tokens):
        """Rejoins tokenized tokens into strings"""
        text = " ".join(tokens)
        cleaned = re.sub(r"( +)(\W)", r"\2", text)
        return cleaned

class SimpleCharacterTokenizer(Tokenizer):
    """Tokenizer for turning a string into characters"""

    def _get_default_specials(self):
        return [
            "\x02", # beginning of string (ASCII "Start of text")
            0,
            "\x03", # end of string (ASCII "End of text")
            1,
        ]

    def tokenize(self, data):
        """Tokenizes string into individual characters"""
        tokenized = list(data)
        return tokenized
        
    def decode_cleanup(self, tokens):
        """Rejoins individual characters into a string"""
        return "".join(tokens)

