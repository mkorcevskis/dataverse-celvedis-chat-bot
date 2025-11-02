import os
import shutil
import time
import urllib.parse as up
import requests
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage-old")
HEADERS = {"User-Agent": "dataverse-celvedis-chat-bot/0.1 (edu demo)"}


def fetch(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=25)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            html = r.text
            return html, BeautifulSoup(html, "html.parser")
    except:
        return None, None
    return None, None


def canonical(soup: BeautifulSoup, fallback: str) -> str:
    link = soup.find("link", rel=lambda v: v and "canonical" in v)
    href = link["href"].strip() if link and link.get("href") else fallback
    p = up.urlparse(href)
    path = p.path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return up.urlunparse(p._replace(path=path))


def html_to_text_and_meta(url: str, soup: BeautifulSoup) -> Tuple[str, Dict]:
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    for sel in ["header", "footer", ".fusion-footer", ".fusion-header", ".site-footer"]:
        for x in soup.select(sel):
            x.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    for hx in ["h1", "h2", "h3"]:
        for h in soup.find_all(hx):
            h.string = f"\n\n###{hx.upper()}### {h.get_text(strip=True)}\n\n"
    text = soup.get_text(separator="\n", strip=True)
    return text, {"source": url, "title": title}


def heading_blocks(text: str) -> List[str]:
    blocks, buf = [], []
    for ln in text.splitlines():
        if ln.startswith("###H1###") or ln.startswith("###H2###") or ln.startswith("###H3###"):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        buf.append(ln)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b and len(b) > 100]


def load_urls(path="urls.txt") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    out = []
    for u in urls:
        p = up.urlparse(u)
        path = p.path
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        out.append(up.urlunparse(p._replace(path=path)))
    return list(dict.fromkeys(out))


def main():
    urls = load_urls("urls.txt")
    if not urls:
        print("urls.txt is empty")
        return

    docs: List[Document] = []
    urls_width = len(str(len(urls)))
    for i, url in enumerate(urls, 1):
        html, soup = fetch(url)
        if not soup:
            print(f"[skip] {url}")
            continue
        canon = canonical(soup, url)
        text, meta = html_to_text_and_meta(canon, soup)
        if text and len(text) > 200:
            blocks = heading_blocks(text) or [text]
            for b in blocks:
                docs.append(Document(page_content=b, metadata=meta))
            print(f"[{i:0{urls_width}}/{len(urls)}] OK: {canon} (+{len(blocks)} {"block" if len(blocks) == 1 else "blocks"})")
        time.sleep(0.2)

    if not docs:
        print("No documents found")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n###H3###", "\n###H2###", "\n###H1###", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    client = PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("celvedis")
    except Exception:
        pass

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="celvedis",
        client=client
    )

    print(f"Indexed: pages={len(set(d.metadata['source'] for d in docs))}, chunks={len(chunks)}")
    print(f"Saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
