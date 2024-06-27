import re

class Stopper:
    def __call__(self, model, path):
        raise NotImplementedError

class CountStopper(Stopper):
    """Stops generation after a given number of tokens

    Parameters
    ----------
    count_to : int
        number of tokens at which the generation of a sequence will be stopped
    """

    def __init__(self, count_to=100):
        self.count_to = count_to

    def __call__(self, model, path):
        return len(path) >= self.count_to


class TokenIDStopper(Stopper):
    """Stops generation after a particular token occurs

    This is helpful to automatically stop generation when, e.g., an
    end-of-sentence token is reached.

    Parameters
    ----------
    token_id : int
        token ID that causes generation to stop
    """

    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, model, path):
        return path[-1] == self.token_id


class StringMatchStopper(Stopper):
    """Stops generation when the decoded string matches a regular expression

    Given a tokenizer and a regular expression, this stopper stops generation
    when the generated sequence, as decoded by the tokenizer, matches the
    regular expression. Helpful for when you want to stop generating on some
    textual condition that can't easily be expressed as a token ID.

    Parameters
    ----------
    tokenizer
        a shoestring.Tokenizer instance
    pattern : str
        the regular expression to match
    """

    def __init__(self, tokenizer, pattern):
        self.tokenizer = tokenizer
        self.pattern = pattern

    def __call__(self, model, path):
        result = re.search(self.pattern, self.tokenizer.decode_one(path))
        return result

