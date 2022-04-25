import spacy


def largest_containing_nounphrase(node: spacy.tokens.span.Span) -> spacy.tokens.span.Span:
    """
    Retrieves the highest level noun phrase from the given node/span.

    :param node: A node in the constituency parse by benepar.
    :type node: spacy.tokens.span.Span
    :return: A node in the constituency parse by benepar that is the highest level NP (or the original node if not a NP).
    :rtype: spacy.tokens.span.Span
    """
    if len(node._.labels) == 0:
        return node

    prev_np = node
    while node._.labels[0] == "NP":
        prev_np = node
        node = node._.parent
        if node is None:
            break
    return prev_np
