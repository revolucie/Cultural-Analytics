#Keyness 
#!/usr/bin/env python3
"""
Keyness-Analyse NS-Unterhaltungsfilm – TF-IDF + Log-Likelihood nach Phasen
"""
import os, re, math
from collections import Counter
from pathlib import Path
from collections import Counter

import spacy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

FARBEN = {1: "#B5533C", 2: "#3B5998", 3: "#6B8F71"}

# ── Pfad anpassen ────────────────────────────────────────────────────────────
TRANSKRIPT_DIR = r"C:\Users\lucie\OneDrive\Dokumente\Studium\MasterInformatik\Cultural Analytics\NS Film Audio\Transkripte"
OUTPUT_DIR     = os.path.join(TRANSKRIPT_DIR, "Keyness_Analyse")
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Phasenzuordnung (Dateiname-Fragment → Phase) ──────────────────────────────
PHASEN_MAP = {
    # ── Phase 1: 1935–1939 ──
        "Schwarze Rosen": 1,
        "Krach im Hinterhaus": 1,
        "Allotria": 1,
        "Verräter": 1,
        "Truxa": 1,
        "indische Grabmal": 1,
        "Mustergatte": 1,
        "Gasparone": 1,
        "Olympia 1": 1,   
        "Urlaub auf Ehrenwort": 1,
        "Patrioten77m09s": 1,
        "Zu neuen Ufern": 1,
        "Mann, der Sherlock Holmes": 1,
        "La Habanera": 1,
        "Pour le Merite": 1,
        "Heimat (1938)": 1,
        "Blaufuchs (1938)": 1,
        "Geheimzeichen LB 17": 1,
        "Sergeant Berry": 1,
        "13 Stühle": 1,
        "Kautschuk": 1,
        "Mutterliebe (Käthe Dorsch": 1,
        "Es War Eine Rauschende Ballnacht": 1,
        "Robert Koch der Bekämpfer des Todes": 1,
        "Opernball": 1,
        "D III 88": 1,
        "Befreite Hände": 1,
        "Maria Ilona": 1,
        "Paradies der Junggesellen": 1,
        "Lied der Wüste": 1,
        "Reise nach Tilsit": 1,
        "Dreizehn Stühle": 1,

    # ── Phase 2: 1940–1942 ──
    "Wunschkonzert": 2,
    "Jew Suess": 2,
    "Operette (1940)": 2,
    "DerPostmeister1940": 2,
    "Bismarck Spielfilm (1940)": 2,
    "Herz der Konigin (1940)": 2,
    "Ein Leben lang": 2,
    "Geierwally": 2,
    "Rosen in Tirol": 2,
    "Feinde": 2,
    "Wiener Blut (1942)": 2,              
    "Kora Terry": 2,
    "Frauen Sind Doch Bessere Diplomaten": 2,
    "Annelie 1941": 2,
    "Ohm Krüger": 2,
    "Ich klage an": 2,
    "Reitet für Deutschland": 2,
    "Quax, der Bruchpilot": 2,
    "Weg ins Freie": 2,
    "Frau Luna": 2,
    "Die goldene Stadt (1942)": 2,
    "Die Grosse Liebe (1942)": 2,
    "Hab mich Lieb": 2,
    "Fronttheater (Rene Dettgen": 2,
    "Entlassung (1942)": 2,
    "Hochzeit auf Bärenhof": 2,
    "Der große Schatten(Heinrich George": 2,
    "Heimkehr (Paula Wessely": 2,             
    "Der große König-1942": 2,            
    "Tanz mit dem Kaiser (1941)": 2,                      

    # ── Phase 3: 1943–1945 ──
    "weiße Traum - Spielfilm": 3,
    "Münchhausen (1943)": 3,
    "Immensee (1943)": 3,
    "Damals (1943)": 3,
    "Altes Herz wird wieder Jung": 3,
    "Zirkus Renz": 3,
    "Bad auf der Tenne 1943": 3,
    "Tonelli": 3,
    "Gabriele Dambrone": 3,
    "Opfergang.1944": 3,
    "Frau meiner Träume (1944)": 3,
    "Schrammeln (Hans Holt": 3,
    "gebieterische Ruf": 3,
    "Ich brauche dich": 3,
    "Engel mit dem Saitenspiel": 3,
    "Herz muß schweigen": 3,
    "Zaubergeige": 3,
}


PHASEN_LABEL = {
    1: "Phase 1 (1934–39)",
    2: "Phase 2 (1940–42)",
    3: "Phase 3 (1943–45)",
}

# ── Spacy laden ───────────────────────────────────────────────────────────────
print("Lade spaCy-Modell...")
nlp = spacy.load("de_core_news_sm", disable=["parser"])

STOPWORDS_EXTRA = {"ja", "nein", "mal", "schon", "noch", "auch", "nur",
                   "mehr", "sehr", "doch", "nun", "wohl", "ganz", "immer",
                   "halt", "gut", "ah", "oh", "ach", "na", "ne", "bitte", "nee", "bloß", "meter"}

def lade_text(pfad):
    """Liest Transkript, entfernt Zeitstempel wie [00:12 → 00:42]"""
    text = Path(pfad).read_text(encoding="utf-8")
    text = re.sub(r"\[\d+:\d+[\.,]?\d*\s*→\s*\d+:\d+[\.,]?\d*\]", "", text)
    return text.strip()

def verarbeite(text):
    """Lemmatisierung, Stopwörter entfernen, nur Nomen/Verben/Adj"""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token.lemma_) > 2
        and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and not token.ent_type_ in {"PER", "LOC", "ORG"} 
        and token.lemma_.lower() not in STOPWORDS_EXTRA
    ]
    return tokens

# ── Texte einlesen & zuordnen ─────────────────────────────────────────────────
print("Lese Transkripte...")
phase_tokens  = {1: [], 2: [], 3: []}
phase_docs    = {1: [], 2: [], 3: []}  # für TF-IDF: pro Film ein String

film_token_data = []  # Liste mit (Phase, Filmname, Counter(tokens)) für spätere Kontextanalyse

for txt_file in Path(TRANSKRIPT_DIR).glob("*.txt"):
    name = txt_file.stem
    phase = None
    for fragment, p in PHASEN_MAP.items():
        if fragment.lower() in name.lower():
            phase = p
            break
    if phase is None:
        print(f"  ⚠ Nicht zugeordnet: {name}")
        continue

    print(f"  ✔ Phase {phase}: {name}")
    text   = lade_text(txt_file)
    tokens = verarbeite(text)
    phase_tokens[phase].extend(tokens)
    phase_docs[phase].append(" ".join(tokens))  # ← ein Eintrag pro Film, nicht pro Phase

    # Zusätzlich für spätere Suche speichern
    film_token_data.append((phase, name, Counter(tokens)))

# Rohfrequenzen pro Phase
LABELS = {1: "Phase 1 (1935–39)", 2: "Phase 2 (1940–42)", 3: "Phase 3 (1943–45)"}

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for i, phase in enumerate([1, 2, 3]):
    freq = Counter(phase_tokens[phase])          # phasetokens kommt aus deinem Script
    data = freq.most_common(15)[::-1]
    words = [d[0] for d in data]
    counts = [d[1] for d in data]

    ax = axes[i]
    bars = ax.barh(words, counts, color=FARBEN[phase])
    for bar, val in zip(bars, counts):
        ax.text(val + 4, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=8.5)
    ax.set_title(LABELS[phase], fontsize=12, fontweight="bold", color=FARBEN[phase])
    ax.set_xlabel("Absolute Häufigkeit")
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Häufigste Wörter pro Phase (roh)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("haeufigste_woerter_raw.png", dpi=150)


# ── TF-IDF pro Phase ──────────────────────────────────────────────────────────
print("\nBerechne TF-IDF...")
phase_texts = [" ".join(phase_tokens[p]) for p in [1, 2, 3]]

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
matrix = tfidf.fit_transform(phase_texts)
vocab  = tfidf.get_feature_names_out()

tfidf_results = {}
for i, phase in enumerate([1, 2, 3]):
    scores = matrix[i].toarray()[0]
    top_idx = scores.argsort()[::-1][:30]
    tfidf_results[phase] = [(vocab[j], scores[j]) for j in top_idx]

def log_likelihood(a, b, total_a, total_b):
    E1 = total_a * (a + b) / (total_a + total_b)
    E2 = total_b * (a + b) / (total_a + total_b)
    def safe(obs, exp):
        return obs * math.log(obs / exp) if obs > 0 and exp > 0 else 0
    return 2 * (safe(a, E1) + safe(b, E2))

# ── Log-Likelihood (G²) ───────────────────────────────────────────────────────
print("Berechne Log-Likelihood Keyness...")
ll_results = {}

for focal_phase in [1, 2, 3]:
    ref_phases = [p for p in [1, 2, 3] if p != focal_phase]
    focal_tokens = phase_tokens[focal_phase]
    ref_tokens   = [t for p in ref_phases for t in phase_tokens[p]]

    focal_freq = Counter(focal_tokens)
    ref_freq   = Counter(ref_tokens)
    total_f    = len(focal_tokens)
    total_r    = len(ref_tokens)

    # pro Wort zählen, in wie vielen Filmen der Phase es vorkommt
    film_freq_in_phase = Counter()
    for doc_text in phase_docs[focal_phase]:           
        doc_words = set(doc_text.split())
        for word in doc_words:
            film_freq_in_phase[word] += 1

    scores = []
    all_words = set(focal_freq.keys()) | set(ref_freq.keys())
    for word in all_words:
        a = focal_freq.get(word, 0)
        b = ref_freq.get(word, 0)

        # Filter: mind. 5 Vorkommen insgesamt
        if a < 5:
            continue
        # mind. in 3 Filmen der Phase vorhanden
        if film_freq_in_phase.get(word, 0) < 5:
            continue

        ll =log_likelihood(a, b, total_f, total_r)
        if a / total_f > b / total_r:
            scores.append((word, ll, a))

    scores.sort(key=lambda x: -x[1])
    ll_results[focal_phase] = scores[:30]

# ── Frequenzentwicklung ausgewählter Wörter über Phasen ──────────────────────
TRACKING_WORDS = ["leutnant", "majestät", "spielen"]

track_data = {}
for word in TRACKING_WORDS:
    track_data[word] = []
    for phase in [1, 2, 3]:
        freq  = Counter(phase_tokens[phase])
        total = len(phase_tokens[phase]) or 1
        track_data[word].append(freq.get(word, 0) / total * 10000)  # pro 10.000 Tokens

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
x     = [1, 2, 3]
xlabels = [PHASEN_LABEL[p] for p in [1, 2, 3]]
markers = ["o", "s", "^"]

colors = [FARBEN[1], FARBEN[2], FARBEN[3]]

for i, word in enumerate(TRACKING_WORDS):
    ax.plot(x, track_data[word],
            label=word,
            color=FARBEN[i+1],
            marker=markers[i],
            linewidth=2.5,
            markersize=9)
    for xi, yi in zip(x, track_data[word]):
        ax.annotate(f"{yi:.1f}", (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center", fontsize=9, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=11)
ax.set_ylabel("Frequenz pro 10.000 Tokens", fontsize=10)
ax.set_title("Diskurswandel: Häufigkeitsentwicklung ausgewählter Keywords", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "word_trends.png"), dpi=150)
plt.close()
print("✔ word_trends.png gespeichert.")


# ── CSV exportieren ───────────────────────────────────────────────────────────
for phase in [1, 2, 3]:
    df_ll = pd.DataFrame(ll_results[phase], columns=["Wort", "Log-Likelihood (G²)", "Freq"])
    df_ll.to_csv(os.path.join(OUTPUT_DIR, f"keyness_phase{phase}.csv"), index=False, encoding="utf-8")

    df_tf = pd.DataFrame(tfidf_results[phase], columns=["Wort", "TF-IDF-Score"])
    df_tf.to_csv(os.path.join(OUTPUT_DIR, f"tfidf_phase{phase}.csv"), index=False, encoding="utf-8")

print("✔ CSVs gespeichert.")

# ── Visualisierungen ──────────────────────────────────────────────────────────

# 1) Barplots Log-Likelihood Top-20
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for i, phase in enumerate([1, 2, 3]):
    data  = ll_results[phase][:20]
    words = [d[0] for d in data]
    vals  = [d[1] for d in data]
    axes[i].barh(words[::-1], vals[::-1], color=FARBEN[phase])
    axes[i].set_title(f"Top Keywords\n{PHASEN_LABEL[phase]}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Log-Likelihood (G²)")
    axes[i].tick_params(axis="y", labelsize=9)

plt.suptitle("Keyness-Analyse NS-Unterhaltungsfilm (Log-Likelihood)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "keyness_barplot.png"), dpi=150)
plt.close()

# 2) Wordclouds pro Phase
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, phase in enumerate([1, 2, 3]):
    freq_dict = {w: ll for w, ll, _ in ll_results[phase]}
    wc = WordCloud(
        width=600, height=400,
        background_color="white",
        color_func=lambda *a, **kw: FARBEN[phase],
        max_words=50,
        font_path=None,
        prefer_horizontal=0.9
    ).generate_from_frequencies(freq_dict)
    axes[i].imshow(wc, interpolation="bilinear")
    axes[i].axis("off")
    axes[i].set_title(PHASEN_LABEL[phase], fontsize=12, fontweight="bold")

plt.suptitle("Keyword-Wordclouds nach Phase", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds.png"), dpi=150)
plt.close()

#phasenspezifische Stärke der Top-Wörter - wie oft kommt es in dieser Phase vor im Vergleich zu den anderen Phasen? (Frequenzverhältnis)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, phase in enumerate([1, 2, 3]):
    data = []
    for word, ll, freq in ll_results[phase][:10]:
        reffreq  = sum(Counter(phase_tokens[p]).get(word, 0) for p in [1,2,3] if p != phase)
        reftotal = sum(len(phase_tokens[p]) for p in [1,2,3] if p != phase)
        focalfreq    = freq / len(phase_tokens[phase])
        reffreqnorm  = reffreq / (reftotal or 1)
        ratio = focalfreq / (reffreqnorm or 1)
        data.append((word, ratio))
    data.sort(key=lambda x: x[1])          # aufsteigend → höchster Wert oben in barh
    words = [d[0] for d in data]
    vals  = [d[1] for d in data]
    axes[i].barh(words, vals, color=FARBEN[phase])
    axes[i].set_title(f"Top Keywords {PHASEN_LABEL[phase]}\nRelative Stärke vs. andere Phasen", fontsize=11)
    axes[i].set_xlabel('Focal / Referenz Frequenz', fontsize=10)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, phase in enumerate([1, 2, 3]):
    data = []
    for word, ll, freq in ll_results[phase][:10]:
        reffreq  = sum(Counter(phase_tokens[p]).get(word, 0) for p in [1,2,3] if p != phase)
        reftotal = sum(len(phase_tokens[p]) for p in [1,2,3] if p != phase)
        focalfreq    = freq / len(phase_tokens[phase])
        reffreqnorm  = reffreq / (reftotal or 1)
        ratio = focalfreq / (reffreqnorm or 1)
        data.append((word, ratio))
    data.sort(key=lambda x: x[1])        
    words = [d[0] for d in data]
    vals  = [d[1] for d in data]
    axes[i].barh(words, vals, color=FARBEN[phase])
    axes[i].set_title(f"Top Keywords {PHASEN_LABEL[phase]}\nRelative Stärke vs. andere Phasen", fontsize=11)
    axes[i].set_xlabel('Focal / Referenz Frequenz', fontsize=10)
    axes[i].tick_params(axis='y', labelsize=9)
plt.suptitle('Phasenspezifische Keyword-Stärke', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'keyword_strength.png'), dpi=150)
plt.close()


#Kontextsuche zur Prüfung von Verteilungen bestimmter Wörter in den Transkripten

SUCHWORT = "majestät"

print(f"\nKontextsuche: '{SUCHWORT}' in Phase 1\n")
for p, filmname, counter in film_token_data:
    if p != 1 or counter.get(SUCHWORT, 0) == 0:
        continue
    
    # Originaltext nochmal laden und Zeilen mit dem Wort finden
    txt_file = Path(TRANSKRIPT_DIR) / (filmname + ".txt")
    if not txt_file.exists():
        continue
    
    print(f"\n── {filmname} ({counter[SUCHWORT]}×) ──")
    for line in txt_file.read_text(encoding="utf-8").splitlines():
        if SUCHWORT.lower() in line.lower():
            print(f"  {line[:120]}")

        

# Statistik ausgeben
for phase in [1, 2, 3]:
    n_filme = len(phase_docs[phase])
    n_tokens = len(phase_tokens[phase])
    print(f"Phase {phase}: {n_filme} Filme | {n_tokens:,} Tokens")

print(f"Gesamt: {sum(len(phase_tokens[p]) for p in [1,2,3]):,} Tokens")
print(f"Seiten (à 500 Wörter): {sum(len(phase_tokens[p]) for p in [1,2,3]) / 500:.0f}")



print(f"\n✅ Fertig! Alle Ausgaben in: {OUTPUT_DIR}")
print("   keyness_barplot.png, wordclouds.png, heatmap.png")
print("   keyness_phase1/2/3.csv, tfidf_phase1/2/3.csv")

