import re
from langchain_core.documents import Document
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

st.set_page_config(page_title="Regulierungs-Assistent", page_icon="📋", layout="centered")

# --- Eigenes Farbschema (Blau/Gelb/Grün) und etwas Feinschliff per CSS ---
st.markdown("""
<style>
.header-box {
    background-color: #185FA5;
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 1.5rem;
}
.header-box h1 {
    color: #ffffff;
    font-size: 1.3rem;
    margin: 0;
}
.header-box p {
    color: #B5D4F4;
    font-size: 0.85rem;
    margin: 4px 0 0 0;
}
.header-badge {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background-color: #FAC775;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
}
@keyframes lupe-dreht {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.header-badge.laedt span {
    display: inline-block;
    animation: lupe-dreht 1s linear infinite;
}
.antwort-titel {
    font-size: 1.15rem;
    font-weight: 700;
    color: #3B6D11;
    margin-bottom: 0.5rem;
}
div.stButton > button {
    background-color: #378ADD;
    color: white;
    border: none;
    font-weight: 500;
}
div.stButton > button:hover {
    background-color: #185FA5;
    color: white;
}
.hinweis-box {
    background-color: #FAEEDA;
    color: #633806;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    font-size: 0.85rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Ressourcen nur einmal laden (nicht bei jeder Frage neu, nicht bei jedem Nutzer neu)
@st.cache_resource
def lade_system():
    # 1. PDFs einlesen
    dateien = ["GasNEF.pdf", "NEST_Effizienzvergleich_Gas.pdf", "RAMEN_Gas.pdf"]
    alle_seiten = []
    for datei in dateien:
        loader = PyPDFLoader(datei)
        alle_seiten.extend(loader.load())

    # 2. In Chunks teilen (identisch zu den bisherigen Tests)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(alle_seiten)

    # 3. Embeddings + Vektordatenbank aufbauen (im Arbeitsspeicher der Cloud-App,
    #    kein persist_directory nötig, da bei jedem Start neu erzeugt)
    embedding_modell = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vektordb = Chroma.from_documents(chunks, embedding_modell)

    # 4. LLM mit API-Key aus den Streamlit-Secrets
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=st.secrets["GROQ_API_KEY"])

    return vektordb, llm

vektordb, llm = lade_system()

def frage_stellen(frage):
     # Mehr Kandidaten holen und winzige Bruchstücke aussortieren
    kandidaten = vektordb.similarity_search(frage, k=24)
    treffer = [d for d in kandidaten if len(d.page_content) > 150][:8]

    # Abkürzungen aus der Frage (z. B. NEST, RAMEN) zusätzlich wörtlich suchen
    for abk in re.findall(r"\b[A-ZÄÖÜ]{3,}\b", frage):
        wort_treffer = vektordb.get(where_document={"$contains": abk}, limit=3,
                                    include=["documents", "metadatas"])
        for text, meta in zip(wort_treffer["documents"], wort_treffer["metadatas"]):
            if len(text) > 150:
                treffer.append(Document(page_content=text, metadata=meta))
    kontext = "\n\n---\n\n".join([doc.page_content for doc in treffer])
    prompt_text = f"""Beantworte die Frage NUR auf Basis des folgenden Kontexts aus Regulierungsdokumenten. Nutze kein Wissen von außerhalb des Kontexts.
Enthält der Kontext Informationen zur Frage, auch nur teilweise oder verstreut auf mehrere Textstellen, dann beantworte die Frage mit diesen Informationen. Fasse verstreute Angaben zusammen, leite bei Bedarf Bedeutungen aus dem Kontext ab und weise auf Lücken hin. Beginne solche zusammengefassten oder abgeleiteten Antworten mit dem Satz "Aus dem Kontext abgeleitet:".
Antworte nur dann mit dem einen Wort KEINE_INFORMATION und sonst nichts, wenn der Kontext zum Thema der Frage überhaupt nichts Verwertbares enthält.
Andernfalls antworte auf Deutsch.

Kontext:
{kontext}

Frage: {frage}

Antwort:"""
    antwort = llm.invoke(prompt_text)
    return antwort.content, treffer

# --- Oberfläche ---
def zeige_header(laedt=False):
    badge_klasse = "header-badge laedt" if laedt else "header-badge"
    header_platzhalter.markdown(f"""
    <div class="header-box">
        <div>
            <h1>📋 Regulierungs-Dokumenten-Assistent</h1>
            <p>KI-Portfolio-Projekt · RAG-Prototyp zu Gasnetz-Regulierungstexten</p>
        </div>
        <div class="{badge_klasse}"><span>🔍</span></div>
    </div>
    """, unsafe_allow_html=True)

header_platzhalter = st.empty()
zeige_header(laedt=False)

# Vorformulierte Beispielfragen als klickbare Chips
if "frage_eingabe" not in st.session_state:
    st.session_state.frage_eingabe = ""

beispielfragen = ["Adressat der GasNEF-Festlegung?", "Rechtsgrundlage EnWG?", "Was ist RAMEN?"]
st.caption("Beispiele zum Ausprobieren:")
spalten = st.columns(len(beispielfragen))
for spalte, beispiel in zip(spalten, beispielfragen):
    if spalte.button(beispiel, use_container_width=True):
        st.session_state.frage_eingabe = beispiel

frage = st.text_input("Stelle eine Frage zu den Regulierungsdokumenten:", key="frage_eingabe")

if st.button("Frage stellen") and frage:
    zeige_header(laedt=True)
    with st.spinner("Suche relevante Textstellen und formuliere Antwort..."):
        antwort, quellen = frage_stellen(frage)
    zeige_header(laedt=False)

    if antwort.strip().startswith("KEINE_INFORMATION"):
        st.info("Dazu finde ich in den Dokumenten keine Angabe.")
    else:
        st.markdown('<p class="antwort-titel">✅ Antwort</p>', unsafe_allow_html=True)
        st.write(antwort)

        st.subheader("Quellen")
        for i, q in enumerate(quellen, 1):
            st.markdown(f"**{i}.** Seite {q.metadata.get('page', '?')} aus `{q.metadata.get('source', '?')}`")
            with st.expander("Textausschnitt anzeigen"):
                st.text(q.page_content)

st.markdown(
    '<div class="hinweis-box">ℹ️ Antworten stammen ausschließlich aus den hochgeladenen Dokumenten und können unvollständig sein.</div>',
    unsafe_allow_html=True
)
