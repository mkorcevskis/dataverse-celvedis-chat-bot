import os
import re
import streamlit as st
from dotenv import load_dotenv
from chromadb import PersistentClient
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "storage-old")
OLLAMA_USE = os.getenv("OLLAMA_USE", "false").lower() in ("true", "1", "on")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

st.set_page_config(page_title="Dataverse ceļveža tērzēšanas bots", layout="wide")
st.title("Dataverse ceļveža tērzēšanas bots")
st.caption("Uzdod jautājumu par \"dataverse.lv\" pētniecības datu pārvaldības ceļveža saturu!")


@st.cache_resource(show_spinner=False)
def load_vs():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = PersistentClient(path=CHROMA_DIR)
    vs = Chroma(
        client=client,
        collection_name="celvedis",
        embedding_function=emb
    )
    return vs


@st.cache_resource(show_spinner=False)
def load_ollama():
    if OLLAMA_USE and HAS_OLLAMA:
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return None


vs = load_vs()
llm = load_ollama()

SYSTEM = """
Tu esi palīgs, kurš atbild tikai latviešu valodā un tikai izmantojot doto kontekstu.
Ja atbilde nav atrodama kontekstā, skaidri pasaki: "Diemžēl atbildi uz šo jautājumu neatradu ceļveža saturā."
Neko neizdomā un neizmanto ārējos avotus. Atbildi īsi (3–6 teikumi).
"""

TEMPLATE = """
{system}

Jautājums (latviešu valodā):
{question}

Konteksts (ceļvedis):
{context}
"""

prompt = PromptTemplate(
    input_variables=["system", "question", "context"],
    template=TEMPLATE
)


def retrieve(q: str, k: int = 5):
    docs = vs.similarity_search(q, k=k)
    ctx = "\n\n".join(d.page_content for d in docs)
    return docs, ctx


def sources_block(docs):
    seen = set()
    out = []
    for d in docs:
        url = d.metadata.get("source", "")
        title = d.metadata.get("title", "") or url
        if url and url not in seen:
            seen.add(url)
            out.append(f"- {title} — {url}")
        if len(out) >= 3:
            break
    return "\n".join(out)


def simple_answer_lv(question: str, docs, ctx: str) -> str:
    sentences = []
    for d in docs[:3]:
        text = d.page_content
        split = re.split(r'(?<=[\.\!\?])\s+', text)
        sentences.extend(split[:2])
        if len(sentences) >= 6:
            break
    body = " ".join(sentences)[:900]
    if not body:
        body = "Diemžēl atbildi uz šo jautājumu neatradu ceļveža saturā."
    src = sources_block(docs)
    return body + ("\n\n**Avoti**\n" + src if src else "")


with st.form("qa"):
    q = st.text_input("Jautājums par Dataverse ceļvedi:", placeholder="Piemēram, kā piešķirt DOI datu kopai?")
    submitted = st.form_submit_button("Jautāt")
    if submitted and q.strip():
        with st.spinner("Meklēju kontekstu..."):
            docs, ctx = retrieve(q, k=5)
        if llm is not None:
            with st.spinner("Veidoju atbildi lokāli, izmantojot Ollama..."):
                final_prompt = prompt.format(system=SYSTEM, question=q, context=ctx or "—")
                answer = llm.invoke(final_prompt).content
            st.markdown("### Atbilde")
            st.write(answer)
        else:
            st.markdown("### Atbilde (bez LLM)")
            st.write(simple_answer_lv(q, docs, ctx))
        st.markdown("**Avoti**")
        st.markdown(sources_block(docs) or "—")
