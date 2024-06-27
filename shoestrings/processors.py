import re
import numpy as np

class Processor:
    def __call__(self, model, path):
        raise NotImplementedError

class StringMatchBoost(Processor):
    """Boosts the score of strings matching a regular expression

    Primarily useful for text generation. If the tokenizer-decoded continuation
    (or entire generated string so far) matches the given regular expression,
    its score is boosted by the amount given in the `match_boost` parameter;
    scores of non-matches are boosted by the amount in the `nonmatch_boost`
    parameter. (These values can be positive or negative.)

    Parameters
    ----------
    tokenizer
        a shoestring.Tokenizer instance
    pattern : str
        the regular expression to match
    match_boost : int or float, optional
        value to add to log probability of matching items
    nonmatch_boost : int or float, optional
        value to add to the log probability of non-matching items
    check_path : boolean, optional
        if False, match against the decoded string of the continuation only; if
        True, check the entire generated string so far
    """

    def __init__(self, tokenizer, pattern, match_boost=2, nonmatch_boost=-10,
                 check_path=False):
        self.tokenizer = tokenizer
        self.pattern = pattern
        self.match_boost = match_boost
        self.nonmatch_boost = nonmatch_boost
        self.check_path = check_path

    def __call__(self, model, path, logprobs):
        modified = []
        for cid, prob in logprobs:
            to_decode = path + (cid,) if self.check_path else (cid,)
            decoded = self.tokenizer.decode_one(to_decode)
            if re.search(self.pattern, decoded):
                modified.append((cid, prob + self.match_boost))
            else:
                modified.append((cid, prob + self.nonmatch_boost))
        return modified

class RepeatBoost(Processor):
    """Boosts the score of continuations already present in generated tokens

    If the token ID of the continuation is already present in the generated
    list of token IDs, the value of `boost` is added to the score of the
    continuation in question. This booster can be used to encourage repeated
    tokens (with a positive `boost` parameter) or discourage them (with a
    negative `boost` parameter).

    Parameters
    ----------
    boost : int or float, optional
        amount to boost log probability of repeated continuations
    """
    def __init__(self, boost=2):
        self.boost = boost

    def __call__(self, model, path, logprobs):
        return [(cid, prob + self.boost if cid in path else prob)
                for cid, prob in logprobs]


class TokenIDBoost(Processor):
    """Boosts the score of continuations matching a list of token IDs

    Each continuation is checked against the list of supplied tokens. If the
    continuation's token ID matches any of these, the log probability score of
    the continuation is boosted by the given amount.

    Parameters
    ----------
    token_ids : list
        list of token IDs to check
    boost : int or float, optional
        value to add to continuations whose token ID is in supplied list
    """

    def __init__(self, token_ids, boost=2):
        self.tokens = set(token_ids)
        self.boost = boost

    def __call__(self, model, path, logprobs):
        return [(cid, prob + self.boost if cid in self.tokens else prob)
                for cid, prob in logprobs]


class AlwaysPickToken(Processor):
    """Always pick the given token, if available

    Causes a given token to *always* be selected, if it is among the possible
    continuations at a given timestep. This behavior is potentially helpful for
    bringing generated sequences to elegant ends. The `after_length` parameter
    sets the minimum length a sequence must reach before the forced token
    selection begins.

    Parameters
    ----------
    token_id : int
        token ID of the token to always pick
    after_length : int
        number of tokens in generated sequence to wait until picking
    """

    def __init__(self, token_id, after_length=10):
        self.token = token_id
        self.after_length = after_length

    def __call__(self, model, path, logprobs):
        if (self.token in [item[0] for item in logprobs]
                and len(path) > self.after_length):
            modified = []
            for cid, prob in logprobs:
                if cid == self.token:
                    modified.append((cid, 0))
            return modified
        else:
            return logprobs

class StringCallbackBoost(Processor):
    """Invokes a callable with decoded string for each continuation

    Given a tokenizer and a callable, invokes the callable once for each
    continuation. The callable is provided with the decoded continuation,
    the decoded generated string so far, and the pre-existing log probability
    score. The callable should return a modified log probability score for the
    continuation.

    The purpose of this class is to make it a bit easier to provide a custom
    scorer: the user only needs to supply a function or lambda that evaluates
    an expression for each decoded continuation (rather than decoding the
    continuation and building a list of probabilities).

    Parameters
    ----------
    tokenizer
        a tokenizer object (e.g., an instance of `shoestrings.Tokenizer`)
    callback : callable
        a callable, to be invoked for each continuation
    """

    def __init__(self, tokenizer, callback):
        self.tokenizer = tokenizer
        self.callback = callback

    def __call__(self, model, path, logprobs):
        modified = []
        for cid, prob in logprobs:
            modified.append((cid, self.callback(
                self.tokenizer.decode_one([cid]),
                self.tokenizer.decode_one(path),
                prob)))
        return modified

