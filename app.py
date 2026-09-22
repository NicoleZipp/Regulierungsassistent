import re
from langchain_core.documents import Document
import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

st.set_page_config(page_title="Regulierungs-Assistent", page_icon="📋")

# Ressourcen nur einmal laden (nicht bei jeder Frage neu)
@st.cache_resource
def lade_system():
    embedding_modell = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vektordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_modell)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
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
st.title("📋 Regulierungs-Dokumenten-Assistent")
st.caption("KI-Portfolio-Projekt · RAG-Prototyp zu Gasnetz-Regulierungstexten")

frage = st.text_input("Stelle eine Frage zu den Regulierungsdokumenten:")

if st.button("Frage stellen") and frage:
    with st.spinner("Suche relevante Textstellen und formuliere Antwort..."):
        antwort, quellen = frage_stellen(frage)

    if antwort.strip().startswith("KEINE_INFORMATION"):
        st.info("Dazu finde ich in den Dokumenten keine Angabe.")
    else:
        st.subheader("Antwort")
        st.write(antwort)

        st.subheader("Quellen")
        for i, q in enumerate(quellen, 1):
            st.markdown(f"**{i}.** Seite {q.metadata.get('page', '?')} aus `{q.metadata.get('source', '?')}`")
            with st.expander("Textausschnitt anzeigen"):
                st.text(q.page_content)
