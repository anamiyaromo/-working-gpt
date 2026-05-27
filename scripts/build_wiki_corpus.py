import argparse
import bz2
import html
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


KOWIKI_LATEST_URL = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2"
DEFAULT_TARGET_CHARS = 1_000_000


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"using existing file: {output_path}")
        return
    if output_path.exists() and output_path.stat().st_size == 0:
        output_path.unlink()

    print(f"downloading: {url}")
    print(f"to: {output_path}")
    sys.stdout.flush()
    with urllib.request.urlopen(url, timeout=60) as response, output_path.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0"))
        seen = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            seen += len(chunk)
            if total:
                pct = 100 * seen / total
                print(f"\rdownloaded {seen / 1e9:.2f} GB / {total / 1e9:.2f} GB ({pct:.1f}%)", end="")
            else:
                print(f"\rdownloaded {seen / 1e9:.2f} GB", end="")
    print()


def clean_wiki_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[File:[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Image:[^\]]+\]\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^\s*[\*\#;:]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_text_block(out, text: str, remaining_chars: int) -> int:
    if not text:
        return 0
    text = text[:remaining_chars]
    out.write(text)
    out.write("\n\n")
    return len(text)


def iter_kowiki_articles(dump_path: Path):
    namespace = ""
    with bz2.open(dump_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("start", "end")):
            if event == "start" and not namespace:
                match = re.match(r"\{(.+)\}", elem.tag)
                namespace = match.group(1) if match else ""
            if event != "end" or not elem.tag.endswith("page"):
                continue

            ns = elem.findtext(f"{{{namespace}}}ns") if namespace else elem.findtext("ns")
            text_elem = elem.find(f".//{{{namespace}}}text") if namespace else elem.find(".//text")
            if ns == "0" and text_elem is not None and text_elem.text:
                yield clean_wiki_text(text_elem.text)
            elem.clear()


def iter_plain_text(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buffer = []
        for line in f:
            line = line.strip()
            if not line:
                if buffer:
                    yield clean_wiki_text("\n".join(buffer))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            yield clean_wiki_text("\n".join(buffer))


def append_source(out, source_name: str, articles, target_chars: int, written: int) -> int:
    print(f"processing source: {source_name}")
    for article in articles:
        if written >= target_chars:
            break
        written += write_text_block(out, article, target_chars - written)
        if written and written % 5_000_000 < 10_000:
            print(f"written about {written:,} chars")
    print(f"source done: {source_name}, total chars: {written:,}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Korean wiki corpus for TinyGPT training.")
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--output", default="data/raw/wiki_ko_namu_1m.txt")
    parser.add_argument("--cache-dir", default="data/external")
    parser.add_argument("--skip-kowiki", action="store_true")
    parser.add_argument("--kowiki-url", default=KOWIKI_LATEST_URL)
    parser.add_argument("--kowiki-dump", default=None)
    parser.add_argument("--namuwiki-text", default=None, help="Optional local Namuwiki plain-text file.")
    parser.add_argument("--dry-run-chars", type=int, default=None, help="Override target size for a quick test.")
    args = parser.parse_args()

    target_chars = args.dry_run_chars or args.target_chars
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        if args.namuwiki_text:
            namu_path = Path(args.namuwiki_text)
            if not namu_path.exists():
                raise FileNotFoundError(f"Namuwiki text file not found: {namu_path}")
            written = append_source(out, "namuwiki", iter_plain_text(namu_path), target_chars, written)

        if not args.skip_kowiki and written < target_chars:
            if args.kowiki_dump:
                dump_path = Path(args.kowiki_dump)
            else:
                dump_path = Path(args.cache_dir) / "kowiki-latest-pages-articles.xml.bz2"
                download_file(args.kowiki_url, dump_path)
            written = append_source(out, "kowiki", iter_kowiki_articles(dump_path), target_chars, written)

    print(f"corpus saved: {output_path}")
    print(f"total chars: {written:,}")
    if written < target_chars:
        print(f"warning: target was {target_chars:,}, but only {written:,} chars were written", file=sys.stderr)


if __name__ == "__main__":
    main()
