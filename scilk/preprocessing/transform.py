from functools import reduce
from typing import Iterable, Tuple, Pattern, Union, Text, Callable


Transform = Callable[[Text], Text]


def ptransform(transformations: Iterable[Tuple[Pattern, Union[Text, Callable]]],
               text: Text) -> Text:
    """
    Pattern transform. The patterns are applied in iteration order with no
    intermediate masking.
    :param transformations: pairs of patterns and replacements (refer to
    `re.sub`'s documentation for more information on possible replacements);
    :param text: text to transform
    :return: transformed text
    """
    return reduce(lambda s, t: t[0].sub(t[1], s), transformations, text)


if __name__ == "__main__":
    raise RuntimeError
