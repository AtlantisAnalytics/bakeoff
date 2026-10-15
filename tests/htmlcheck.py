"""Tag-balance check for generated HTML pages (shared by the page, build and feed tests)."""

from html.parser import HTMLParser

VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split(" "))


class _Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in VOID:
            self.errors.append(f"end tag for void element </{tag}>")
        elif not self.stack or self.stack[-1] != tag:
            top = self.stack[-1] if self.stack else None
            self.errors.append(f"unexpected </{tag}> (open: {top})")
        else:
            self.stack.pop()


def assert_balanced(html: str) -> None:
    p = _Balance()
    p.feed(html)
    p.close()
    assert not p.errors, p.errors
    assert not p.stack, f"unclosed tags: {p.stack}"
