def indent_string(text: str, indent: str = "\t") -> str:
    """
    Indents every line in a multi-line string with the given indent string.

    Parameters:
    -----------
    text : str
        The input string, possibly containing multiple lines separated by '\n'.

    indent : str
        The string to prepend to each line (default: tab character).

    Returns:
    --------
    str
        A new string where every line is prefixed with `indent`.
    """
    return "\n".join(f"{indent}{line}" for line in text.splitlines())
