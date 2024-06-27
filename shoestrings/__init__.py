from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import softmax
import numpy as np
import random

from shoestrings.stoppers import CountStopper, TokenIDStopper
from shoestrings.processors import AlwaysPickToken, StringCallbackBoost

def ngrams(n, tokens):
    """yields ngrams of length `n` from iterator `tokens`"""
    tokens = list(tokens)
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

class MarkovModel:
    """Implements a Markov model with the given data

    This class provides general facilities for implementing Markov models for
    data generation. For text-specific functions, you may prefer the
    TextGenerator class, which wraps the MarkovModel class and provides some
    niceties for parsing and generating strings.

    The ngram length corresponds to the total length of the context plus
    the continuation, e.g., n=3 builds a model whose contexts have length
    two. (Continuations are always length 1.)

    Parameters
    ----------
    n : int, optional
        the length of the model's ngrams

    Attributes
    ----------
    n : int
        the length of the model's ngrams
    matrix : scipy.sparse.csr_matrix
        a sparse matrix mapping contexts to continuations
    contexts : list
        a list of all contexts in the model
    context2idx : dict
        maps contexts to their index in `matrix`

    """

    def __init__(self, n=3):
        self.n = n

    def build(self, token_id_iter):
        """Builds the Markov model with the given token IDs

        You can supply token IDs using any iterator (lists, generators,
        etc.) Each element of the iterator should itself be an iterator with
        token IDs. (This is to make it easy to read tokens without loading the
        entire source into memory, e.g., one line at a time.)

        Parameters
        ----------
        token_id_iter
            an iterator yielding iterators of token IDs

        """

        all_contexts = []
        all_continuations = []

        for ids in token_id_iter:
            # extract ngrams
            all_contexts.extend([item[:-1] for item in ngrams(self.n, ids)])
            all_continuations.extend(
                    [item[-1] for item in ngrams(self.n, ids)])

        self.contexts = sorted(list(set(all_contexts)))
        self.context2idx = {item: i for i, item in enumerate(self.contexts)}

        # make sparse matrix
        rows = []
        cols = []
        data = []
        for context, continuation in zip(all_contexts, all_continuations):
            rows.append(self.context2idx[context])
            cols.append(continuation)
            data.append(1)

        self.matrix = csr_matrix(coo_matrix((data, (rows, cols))))

    def build_from_one(self, token_ids):
        """Build from a flat list of token IDs

        This is a convenience function for building from a flat list of IDs
        (rather than nested iterators, as required by `.build()`).

        Parameters
        ----------
        token_ids
            iterator of token IDs
        """

        self.build([token_ids])

    def generate_with_probs(self, temperature=1.0, beam_width=1, start=None,
                 processors=None, stoppers=None):
        """Generate chains of tokens, starting with a particular context

        Generates chains of tokens from the model using the given parameters.
        The method uses a beam search algorithm to pick the sequences with the
        highest scores, and a softmax calculation to bias the prediction
        probabilities at each step. Supplying `beam_width=1` and a low
        temperature (e.g., `0.0001`) is equivalent to a greedy search.

        Processors are called compositionally, i.e., the result of the first
        processor is passed as input to the next, and so on.

        Stoppers are called iteratively; if any stopper returns `True`,
        generation is halted for that sequence. If no stopper object is
        supplied as an argument, the model will stop generating after
        100 iterations. (This is to prevent the model from generating new
        tokens forever as a default behavior.)

        If no starting context is provided, a random context will be selected.

        Results are returned in descending order by score (i.e., most likely
        generated sequences are first).

        Parameters
        ----------
        temperature : float, optional
            temperature parameter for softmax calculation
        beam_width : int, optional
            number of possibilities to consider at each step
        start : tuple, optional
            context to use as the start of the chain
        processors : list, optional
            functions or shoestrings.Processor instances for processing
            probabilities
        stoppers : list, optional
            functions or shoestrings.Stopper instances for determining when to
            stop iterating

        Returns
        -------
        list
            tuples of (generated_ids, score)

        """

        if stoppers is None:
            stoppers = [CountStopper(100)]

        if processors is None:
            processors = []

        if start is None:
            start = random.choice(self.contexts)

        completed = []

        # each state has the path plus cumulative log probability
        states = [(start, 0.0)] * beam_width

        # beam search algorithm adapted from Jurafsky & Martin's _Speech and
        # Language Processing_, chapter 13 (2023 edition). but if there's a
        # mistake it's mine.

        # beam search with a beam_width of 1 and a very low temperature is
        # essentially identical to greedy search

        # at every timestep, we need to calculate the cumulative
        # probability of each beam. we can do this by adding the log
        # probability to the previous log probability (because log(x*y) =
        # log(x)+log(y)), then normalizing by length

        while beam_width > 0: # until we've reached a stop for all beams

            extended_frontier = []

            for i, (state_path, state_prob) in enumerate(states):

                context = state_path[-(self.n-1):]
                current = self.context2idx[tuple(context)]
                counts = self.matrix[current]

                logprobs = compute_sparse_logprobs(counts)
                logprobs = process_logprobs(self, processors, state_path,
                                            logprobs)

                # add a new state for each probability and token, adding
                # the probability of this state to the path's cumulative
                # probability
                extended_frontier.extend(
                    [(state_path+(tok,), p+state_prob) for tok, p in logprobs])

            # extended frontier now has an entry for each possible next state.

            # normalize probabilities by length of path
            # FIXME: this lil formula works empirically to gently penalize
            # longer sequences but... i feel like it could be more, like,
            # rigorously motivated. with MATH
            all_probs = [item[1] - len(item[0]) for item in extended_frontier]

            # pick indices according to softmax w/temperature
            picked_indices = log_softmax_with_temperature(
                    np.array(all_probs), temperature, beam_width)

            # overwrite states with picked items from the frontier
            states = [extended_frontier[idx] for idx in picked_indices]

            to_remove = set()

            # check if any of these states are complete
            for i, state in enumerate(states):
                found = False
                for stopper in stoppers:
                    if stopper(self, state[0]):
                        found = True
                        break
                if found:
                    beam_width -= 1
                    completed.append(state[:])
                    to_remove.add(i)

            # filter out completed states
            states = [item for i, item in enumerate(states)
                      if i not in to_remove]

        # return alternatives sorted in reverse order by log probability
        return sorted(completed, reverse=True, key=lambda x: x[1]/len(x[0]))

    def generate(self, temperature=1.0, beam_width=1, start=None,
                     processors=None, stoppers=None):
        """Generate a chain of tokens

        This is a convenience wrapper for `.generate_with_probs()` which
        returns only the generated sequences, rather than tuples with the
        sequences and their scores.

        Parameters
        ----------
        temperature : float, optional
            temperature parameter for softmax calculation
        beam_width : int, optional
            number of possibilities to consider at each step
        start : tuple, optional
            context to use as the start of the chain
        processors : list, optional
            functions or shoestrings.Processor instances for processing
            probabilities
        stoppers : list, optional
            functions or shoestrings.Stopper instances for determining when to
            stop iterating

        Returns
        -------
        list
            each element is a list of generated token IDs

        """

        return [item[0] for item in self.generate_with_probs(temperature,
            beam_width, start, processors, stoppers)]

    def generate_one(self, temperature=1.0, beam_width=1, start=None,
                     processors=None, stoppers=None):
        """Generate a chain of tokens, starting with a particular context

        This is a convenience wrapper for `.generate_with_probs()` that
        generates a single sequence of tokens from the model.

        Parameters
        ----------
        temperature : float, optional
            temperature parameter for softmax calculation
        beam_width : int, optional
            number of possibilities to consider at each step
        start : tuple, optional
            context to use as the start of the chain
        processors : list, optional
            functions or shoestrings.Processor instances for processing
            probabilities
        stoppers : list, optional
            functions or shoestrings.Stopper instances for determining when to
            stop iterating


        Returns
        -------
        list
            a list of generated token IDs
        """

        return self.generate(temperature, beam_width, start, processors,
                             stoppers)[0]


class TextGenerator:
    """An easy-to-use Markov chain text generator

    This class wraps the functionality of `MarkovModel`, providing conveniences
    specifically for text generation.

    Parameters
    ----------
    n : int
        the length of the model's ngrams
    tokenizer
        a shoestrings.Tokenizer instance for tokenizing the input
    source
        a list or iterator of strings, used to build the model (will be
        tokenized with the given tokenizer beforehand)

    Attributes
    ----------
    n : int
        the length of the model's ngrams
    tokenizer
        the shoestrings.Tokenizer instance used to tokenize the input
    model
        the underlying `MarkovModel`
    starts : list
        a list of contexts that occurred at the beginning of any item in the
        source data
    """
        
    def __init__(self, n, tokenizer, source):

        # if source is a generator, we need to use it twice (once to build
        # the tokenizer, and again to build the model)
        cached = list(source)

        self.n = n
        self.tokenizer = tokenizer
        self.tokenizer.build(cached)
        self.model = MarkovModel(self.n)
        self.model.build(self.tokenizer.encode(cached))
        self.starts = [item for item in self.model.contexts
                       if item[0] == tokenizer.BOS_ID]

    def generate(self, temperature=1.0, beam_width=1, start_string=None,
                 max_tokens=100, custom_scorer=None, processors=None,
                 stoppers=None):
        """Generates strings from the model

        This method generates a sequence of tokens using the underlying
        `MarkovModel`, and then decodes them as strings using the
        `TextGenerator`'s tokenizer. Sequences will automatically terminate
        when the tokenizer's `EOS_ID` is reached.

        The callable `custom_scorer`, if supplied, will be called for each
        context and continuation that the model evaluates, with three
        parameters: the tokenizer-decoded string of the continuation, the
        tokenizer-decoded string of the generated string so far, and the
        original log probability of the continuation as assigned by the model.
        The custom scorer should return a modified log probability of the
        continuation.

        Parameters
        ----------
        temperature : float, optional
            temperature parameter for softmax
        beam_width : int, optional
            number of possibilities to consider at each step
        start_string : str, optional
            seed string for generation; must tokenize into a valid starting
            sequence
        max_tokens : int, optional
            the maximum number of tokens to generate in any string
        custom_scorer : callable, optional
            called on each possible prediction to retrieve a custom score
        processors : list, optional
            a list of probability processors
        stoppers : list, optional
            a list of stoppers

        Returns
        -------
        list
            a list of strings generated by the model
        """

        if stoppers is None:
            stoppers = []

        stoppers = stoppers + [TokenIDStopper(self.tokenizer.EOS_ID),
                        CountStopper(max_tokens)]

        if processors is None:
            processors = []

        if custom_scorer is not None:
            processors = [StringCallbackBoost(self.tokenizer,
                                             custom_scorer)] + processors

        if start_string is None:
            start = random.choice(self.starts)
        else:
            start = tuple(self.tokenizer.encode_one(start_string,
                                              add_special=False))

        return self.tokenizer.decode(
                self.model.generate(temperature, beam_width, start=start,
                processors=processors, stoppers=stoppers))

    def generate_one(self, temperature=1.0, beam_width=1, start_string=None,
                     max_tokens=100, custom_scorer=None, processors=None,
                     stoppers=None):
        """Generates a single string from the model

        This is a convenience wrapper for `.generate()` that returns only the
        highest-rank string.

        Parameters
        ----------
        temperature : float, optional
            temperature parameter for softmax
        beam_width : int, optional
            number of possibilities to consider at each step
        start_string : str, optional
            seed string for generation; must tokenize into a valid starting
            sequence
        max_tokens : int, optional
            the maximum number of tokens to generate in any string
        custom_scorer : callable, optional
            called on each possible prediction to retrieve a custom score
        processors : list, optional
            a list of probability processors
        stoppers : list, optional
            a list of stoppers

        Returns
        -------
        str
            a string generated by the model
        """
        
        return self.generate(temperature, beam_width, start_string,
                             max_tokens, custom_scorer, processors,
                             stoppers)[0]


def log_softmax_with_temperature(logprobs, temperature, n):

    # add a tiny number to avoid divide-by-zero errors
    probs_temp = (logprobs + 10e-16) / temperature

    # thank you scipy for this implementation that resists overflows
    softmax_scores = softmax(probs_temp)

    # draw an index weighted by the calculated softmax
    picked = np.random.choice(np.arange(softmax_scores.shape[0]),
                              p=softmax_scores, size=n)
    return picked

def compute_sparse_logprobs(counts):

    # convert to probability at next step, based on counts
    probs = counts / counts.sum()

    # convert probs to list of tuples, where element zero is the
    # continuation id and element 1 is the log probability of
    # that continuation
    coo = coo_matrix(probs) 
    indices = list(coo.col)
    probs_only = np.array(coo.data)
    logprobs_only = np.log(probs_only)

    logprobs = list(zip(indices, logprobs_only))

    return logprobs

def process_logprobs(model, processors, state_path, logprobs):

    # adjust probabilities according to processors
    # processors receive aforementioned list of tuples
    for proc in processors:
        logprobs = proc(model, state_path, logprobs)
        # clamp logprobs to zero at each step
        logprobs = [(a, 0 if b > 0 else b) for a, b in logprobs]

    return logprobs

