import re
import hashlib
from urllib.parse import urlparse


# URLs that are noise, not content
_NOISE_URL_PATTERNS = (
    "sitemap", "wp-content", "wp-json", "feed", "xmlrpc",
    ".xml", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".css", ".js", "placeholder",
)


def is_noise_url(url: str) -> bool:
    """Return True if the URL is a sitemap, asset, or other non-content page."""
    path = urlparse(url).path.lower()
    return any(pat in path for pat in _NOISE_URL_PATTERNS)


def clean_content(markdown: str) -> str:
    """Clean and normalize scraped markdown content."""
    # Remove lines that are just a wall of URLs
    lines = markdown.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are >70% URLs / image paths
        url_chars = len(re.findall(r'https?://\S+', stripped))
        if url_chars > 3 and len(stripped) < 500:
            continue
        # Skip raw XML/sitemap lines
        if stripped.startswith("<") and stripped.endswith(">"):
            continue
        clean_lines.append(line)

    markdown = "\n".join(clean_lines)

    # Remove excessive blank lines
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    # Remove HTML comments
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)

    # Collapse whitespace runs (preserve newlines)
    markdown = re.sub(r"[ \t]{2,}", " ", markdown)

    # Remove repetitive navigation link blocks
    markdown = re.sub(r"(\[.*?\]\(.*?\)\s*){5,}", "", markdown)

    return markdown.strip()


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def is_meaningful(content: str, min_chars: int = 150) -> bool:
    """Return True if content has enough real text to be worth embedding."""
    # Strip all URLs and count remaining text
    text_only = re.sub(r'https?://\S+', '', content).strip()
    return len(text_only) >= min_chars
