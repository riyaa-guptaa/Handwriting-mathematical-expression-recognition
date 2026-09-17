
# training/prepare_dataset





import json
import pandas as pd

from config import (
    LABEL_FILE,
    VOCAB_FILE
)


SPECIAL_TOKENS = [
    "<PAD>",
    "<START>",
    "<END>",
    "<UNK>"
]


def tokenize(expression):

    tokens = []

    i = 0

    while i < len(expression):

        character = expression[i]

        if character == "\\":

            j = i + 1

            while (
                j < len(expression)
                and expression[j].isalpha()
            ):

                j += 1

            tokens.append(
                expression[i:j]
            )

            i = j

        elif character.isspace():

            i += 1

        else:

            tokens.append(
                character
            )

            i += 1

    return tokens


def build_vocabulary(expressions):

    vocabulary = set(
        SPECIAL_TOKENS
    )

    for expression in expressions:

        tokens = tokenize(
            expression
        )

        vocabulary.update(
            tokens
        )

    vocabulary = sorted(
        vocabulary
    )

    token_to_id = {
        token: index
        for index, token
        in enumerate(vocabulary)
    }

    id_to_token = {
        str(index): token
        for token, index
        in token_to_id.items()
    }

    return token_to_id, id_to_token


def main():

    data = pd.read_csv(
        LABEL_FILE
    )

    expressions = data[
        "expression"
    ].astype(str).tolist()

    token_to_id, id_to_token = (
        build_vocabulary(
            expressions
        )
    )

    with open(
        VOCAB_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "token_to_id": token_to_id,
                "id_to_token": id_to_token
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "Vocabulary created successfully."
    )

    print(
        "Number of tokens:",
        len(token_to_id)
    )


if __name__ == "__main__":

    main()