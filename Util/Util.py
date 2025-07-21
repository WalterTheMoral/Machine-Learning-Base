def indent_string(text: str, indent: str = "\t") -> str:
    """
    Indents every line in a multi-line string with the given indent string.

    :param text: Text to indent
    :return: Indented text
    """
    return "\n".join(f"{indent}{line}" for line in text.splitlines())
