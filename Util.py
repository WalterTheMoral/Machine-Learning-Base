def indent_lines(lines: list[str], tab: str = "\t", skip_first: bool = True) -> str:
    def tab_all_lines(text: str) -> str:
        return "\n".join(tab + line for line in text.splitlines())

    rest = [tab_all_lines(line) for line in lines[1:]] if skip_first else [tab_all_lines(line) for line in lines]
    return "\n".join([first] + rest) if skip_first else "\n".join(rest)
