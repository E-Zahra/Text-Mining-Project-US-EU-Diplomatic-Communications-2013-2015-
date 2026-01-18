# %% [markdown]
# # BERTopic Hyperparameter Optimization (Supervised Approach)
# 
# In this notebook, I applied BERTopic to perform topic modeling on textual data. First, I used an unsupervised approach to automatically discover latent topics based on semantic similarity between documents.
# 
# Next, I optimized key model hyperparameters to improve topic quality, evaluating the results using topic coherence, topic diversity, and the number of topics. This allowed me to select a configuration that balances interpretability and topic separation.
# 
# Finally, I applied supervised topic modeling by incorporating predefined report sections as labels. This helped generate more structured and interpretable topics aligned with the report structure.
# 
# Overall, the notebook demonstrates a complete topic modeling workflow, from exploration to optimization and evaluation.
# 

# %%
%pip install pandas==2.1.4 pyarrow==14.0.2



# %%
import pandas as pd

path = r"output.csv"
df = pd.read_csv(path)

df.head()


# %%
df.groupby(["year"])["country"].nunique(), df["is_eu"].mean()

# %%
docs_all = df["text"].tolist()
tokenized_docs = [d.split() for d in docs_all]


# %% [markdown]
# UNSUPERVISED TOPIC MODELING (BERTopic)

# %%
%pip uninstall torch torchvision torchaudio -y


# %%
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# %%
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# %%
pip install sentence-transformers


# %%
# Install the required libraries

from sentence_transformers import SentenceTransformer


# Use CPU-only version of PyTorch to avoid DLL initialization errors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
embeddings = embedding_model.encode(
    docs_all,
    show_progress_bar=True
)


# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    stop_words="english",
    min_df=3,
    max_df=0.9,
    ngram_range=(1, 2)
)


# %%
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

umap_model = UMAP(
    n_neighbors=10,
    n_components=5,
    metric="cosine",
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=3,
    min_samples=1,
    metric="euclidean"
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics="auto",        # safety net
    verbose=True
)

# %%
%pip install bertopic==0.16.0

# %%
topics, probs = topic_model.fit_transform(docs_all, embeddings)


# %%
topic_model.get_topic_info().head()


# %%
topic_info = topic_model.get_topic_info().copy()

# Exclude outliers for interpretation
main_topics = topic_info[topic_info["Topic"] != -1].copy()

# Add share
main_topics["share"] = main_topics["Count"] / main_topics["Count"].sum()

main_topics.head(30)[["Topic","Count","share","Name"]]


# %%
valid_topic_ids = [
    t for t in set(topics)
    if t != -1
]

len(valid_topic_ids)


# %%
def extract_topic_words(model, topic_ids, topn=10):
    topic_words = []

    for tid in topic_ids:
        topic = model.get_topic(tid)

        # 🔴 skip empty or invalid topics
        if topic is None:
            continue

        words = [w for w, _ in topic if isinstance(w, str)]

        if len(words) >= 2:  # Gensim needs at least 2 words
            topic_words.append(words[:topn])



    return topic_words


# %%
%pip install gensim

# %%
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

def coherence_cv(tokenized_docs, topic_words):
    dictionary = Dictionary(tokenized_docs)

    # ╓ remove words not in dictionary
    filtered_topics = [
        [w for w in topic if w in dictionary.token2id]
        for topic in topic_words
    ]

    filtered_topics = [t for t in filtered_topics if len(t) >= 2]

    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    cm = CoherenceModel(
        topics=filtered_topics,
        texts=tokenized_docs,
        corpus=corpus,
        dictionary=dictionary,
        coherence="c_v"
    )

    return cm.get_coherence()

# %%

def topic_diversity(topic_words, topk=10):
    all_words = []
    for words in topic_words:
        all_words.extend(words[:topk])
    return len(set(all_words)) / len(all_words)




# %%
topic_words = extract_topic_words(
    topic_model,
    valid_topic_ids,
    topn=10
)

len(topic_words)

# %%
coh = coherence_cv(tokenized_docs, topic_words)
div = topic_diversity(topic_words)


# %% [markdown]
# hyperparametr optimization

# %%
base_params = {
    "n_neighbors": 10,
    "n_components": 5,
    "min_cluster_size": 3,
    "min_samples": 1
}


# %%
configs = [
    {**base_params},  # baseline
    {**base_params, "min_cluster_size": 5},
    {**base_params, "n_neighbors": 15},
]


# %%
def build_model_from_params(params):
    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean"
    )

    model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics="auto",
        verbose=False
    )

    return model


# %%
results = []

for i, params in enumerate(configs):
    print(f"Running config {i+1}: {params}")

    model = build_model_from_params(params)
    topics, _ = model.fit_transform(docs_all, embeddings)

    valid_topic_ids = [t for t in set(topics) if t != -1]

    topic_words = extract_topic_words(model, valid_topic_ids, topn=10)

    if len(topic_words) < 2:
        print("  → skipped (too few valid topics)")
        continue

    coh = coherence_cv(tokenized_docs, topic_words)
    div = topic_diversity(topic_words)

    results.append({
        "config_id": i + 1,
        **params,
        "n_topics": len(topic_words),
        "coherence_c_v": coh,
        "topic_diversity": div
    })


# %%
results_df = pd.DataFrame(results)
results_df


# %%
results_df["score"] = (
    0.7 * results_df["coherence_c_v"] +
    0.3 * results_df["topic_diversity"]
)

results_df.sort_values("score", ascending=False)


# %%
best_params = results_df.sort_values(
    "score", ascending=False
).iloc[0]

best_params = {
    "n_neighbors": int(best_params["n_neighbors"]),
    "n_components": int(best_params["n_components"]),
    "min_cluster_size": int(best_params["min_cluster_size"]),
    "min_samples": int(best_params["min_samples"]),
}


# %%
best_model = build_model_from_params(best_params)
topics, _ = best_model.fit_transform(docs_all, embeddings)


# %%
import matplotlib.pyplot as plt

topic_info = topic_model.get_topic_info()
topic_info = topic_info[topic_info.Topic != -1]

plt.figure()
plt.bar(
    topic_info.Topic.astype(str),
    topic_info.Count
)
plt.xlabel("Topic ID")
plt.ylabel("Number of Documents")
plt.title("Topic Size Distribution")
plt.show()


# %%
best_model.get_topic_info()
for t in range(5):
    print(f"Topic {t}:")
    print(best_model.get_topic(t))
    print()


# %%
best_model.get_topic_info()

# %%
topic_info = best_model.get_topic_info().copy()

# Exclude outliers for interpretation
main_topics = topic_info[topic_info["Topic"] != -1].copy()

# Add share
main_topics["share"] = main_topics["Count"] / main_topics["Count"].sum()

main_topics.head(30)[["Topic","Count","share","Name"]]


# %%
docs = best_model.get_representative_docs(1)

for d in docs[:3]:
    print(d[:300])
    print("----")


# %%
plt.figure()
plt.scatter(
    results_df["n_topics"],
    results_df["coherence_c_v"]
)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence (c_v)")
plt.title("Coherence vs Number of Topics")
plt.show()


# %% [markdown]
# SUPERVISED TOPIC MODELING BY REPORT SECTION

# %%
df["section"].value_counts().head()

# %%
sections = [ "Section 1.", "Section 6.", "Section 2."]

# %%
def run_bertopic_on_subset(docs, embeddings_subset=None):
    topics, _ = topic_model.fit_transform(docs, embeddings_subset)
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]  # remove outliers
    return topic_info


# %%
results_by_section = {}

for sec in sections:
    subset = df[df["section"] == sec]["text"].tolist()

    if len(subset) < 100:
        continue  # skip tiny sections

    topics, _ = topic_model.fit_transform(subset)

    results_by_section[sec] = topic_model.get_topic_info()


# %%
for sec, info in results_by_section.items():
    print(f"\n=== {sec} ===")
    display(info.head(5))


# %%
for sec in results_by_section:
    results_by_section[sec] = results_by_section[sec][
        results_by_section[sec]["Topic"] != -1
    ]

# %%
comparison_rows = []

for sec, info in results_by_section.items():
    for _, row in info.iterrows():
        comparison_rows.append({
            "section": sec,
            "topic_id": row["Topic"],
            "n_documents": row["Count"],
            "top_words": row["Name"]
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.head()


# %%
comparison_df.sort_values(
    ["section", "n_documents"],
    ascending=[True, False]
).head(15)


# %%
topic_counts = comparison_df.groupby("section")["topic_id"].nunique()
topic_counts


# %%
df_topics = df.copy()   # Create a copy of the original DataFrame
df_topics["topic"] = best_model.topics_ # Use the topics from the best_model, which was fitted on all documents
df_topics = df_topics[df_topics["topic"] != -1]

# %%
topic_year = (
    df_topics
    .groupby(["year", "topic"])
    .size()
    .reset_index(name="count")
)

topic_year["share"] = (
    topic_year["count"] /
    topic_year.groupby("year")["count"].transform("sum")
)

# %%
eu_df = df_topics[df_topics["is_eu"] == 1]
non_eu_df = df_topics[df_topics["is_eu"] == 0]

# %%
eu_topics = eu_df["topic"].value_counts(normalize=True)
non_eu_topics = non_eu_df["topic"].value_counts(normalize=True)

# %%
topic_compare = pd.DataFrame({
    "EU": eu_topics,
    "Non_EU": non_eu_topics
}).fillna(0)
topic_compare["difference"] = abs(topic_compare["EU"] - topic_compare["Non_EU"])
distinct_topics = topic_compare.sort_values("difference", ascending=False).head(5)

# %%
# Create topic -> name mapping using top keywords
topic_labels = {
    topic: ", ".join([word for word, _ in best_model.get_topic(topic)[:5]])
    for topic in best_model.get_topics().keys()
    if topic != -1
}
df_topics["topic_name"] = df_topics["topic"].map(topic_labels)

# %%
df_topics['EU_status'] = df_topics['is_eu'].map({True: 'EU', False: 'Non-EU'})

# Group by EU_status and topic_name, and count occurrences
topic_counts_by_region = df_topics.groupby(["EU_status", "topic_name"]).size().reset_index(name='count')

# Calculate the share of each topic within its EU_status group
topic_counts_by_region['share'] = topic_counts_by_region.groupby("EU_status")['count'].transform(lambda x: x / x.sum())

# Now topic_region_pivot can be created from this DataFrame
topic_region_pivot = topic_counts_by_region.pivot(
    index="topic_name",
    columns="EU_status",
    values="share"
).fillna(0)

topic_region_pivot.columns = ["Non-EU", "EU"]

# %%
from collections import Counter

def extract_keywords_from_topic_name(topic_name):
    # "police / authorities / detention / prison"
    return [w.strip() for w in topic_name.split("/")]


# %%
def auto_section_label(section_df, top_k=4):
    counter = Counter()

    for _, row in section_df.iterrows():
        words = extract_keywords_from_topic_name(row["topic_name"])
        # Her chunk = 1 oy (istersen ağırlık ekleyebiliriz)
        for w in words:
            counter[w] += 1

    top_words = [w for w, _ in counter.most_common(top_k)]
    return " / ".join(top_words)


# %%
section_labels_df = (
    df_topics
    .groupby("section")
    .apply(auto_section_label)
    .reset_index()
)

section_labels_df.columns = ["section", "section_label_auto"]

section_labels_df

# %%
import pandas as pd

def get_main_phrase(label):
    """
    Virgülle ayrılmış uzun label içinden
    en baskın (ilk) kısmı alır
    """
    if pd.isna(label):
        return ""
    return label.split(",")[0]


# %%
def short_label_2_words(label):
    main = get_main_phrase(label)
    words = main.split()
    return " ".join(words[:2])


# %%
section_labels_df["section_label_short"] = (
    section_labels_df["section_label_auto"]
    .apply(short_label_2_words)
)


# %%
section_counts = (
    df_topics
    .groupby("section")
    .size()
    .reset_index(name="n_chunks")
    .merge(section_labels_df, on="section", how="left")
    .sort_values("n_chunks")
)

plt.figure(figsize=(9, 6))
plt.barh(
    section_counts["section_label_short"],
    section_counts["n_chunks"]
)

plt.xlabel("Number of Chunks")
plt.title("Distribution of Sections (Auto-labeled, Short)")
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Ultra-short topic label creator
# ----------------------------------
def ultra_short_label(label):
    """
    Converts long topic labels into 1–2 word concept labels.
    Example:
    'corruption, disclosure, public access, financial disclosure'
    -> 'corruption'
    """
    if pd.isna(label):
        return "unknown"

    # split by comma or slash
    tokens = (
        label
        .replace("/", ",")
        .split(",")
    )

    # take first meaningful token
    main = tokens[0].strip()

    # keep at most 2 words
    return " ".join(main.split()[:2])


# ----------------------------------
# 2. Prepare plotting dataframe
# ----------------------------------
plot_df = distinct_topics.copy().reset_index()
plot_df.columns = ["topic", "EU", "Non_EU", "difference"]

# use EXISTING topic_labels
plot_df["topic_label_full"] = plot_df["topic"].map(topic_labels)

# create ultra-short labels (like your example)
plot_df["topic_label_short"] = plot_df["topic_label_full"].apply(
    ultra_short_label
)

# use short labels as index
plot_df = plot_df.set_index("topic_label_short")


# ----------------------------------
# 3. Visualization (clean & compact)
# ----------------------------------
plot_df[["EU", "Non_EU"]].plot(
    kind="barh",
    figsize=(8, 4)
)

plt.xlabel("Normalized Share")
plt.title("Top Topics with Largest EU vs Non-EU Differences")
plt.tight_layout()
plt.show()


# %%
plot_df["diff"] = plot_df["EU"] - plot_df["Non_EU"]

plt.figure(figsize=(9, 4))
plt.bar(
    plot_df.index.astype(str),
    plot_df["diff"]
)

plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Topic")
plt.ylabel("EU − Non-EU Share")
plt.title("Topic Share Difference Between EU and Non-EU")
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

def ultra_short_label(label):
    if pd.isna(label):
        return "unknown"
    tokens = label.replace("/", ",").split(",")
    main = tokens[0].strip()
    return " ".join(main.split()[:2])

df_temp = df_topics.copy()

df_temp["topic_label_short"] = (
    df_temp["topic_name"]
    .apply(ultra_short_label)
)

# optional: remove outliers
df_temp = df_temp[df_temp["topic"] != -1]



topic_year_counts = df_temp.groupby(["year", "topic_label_short"]).size().reset_index(name="count")
topic_year_counts["share"] = topic_year_counts["count"] / topic_year_counts.groupby("year")["count"].transform("sum")
topic_year = topic_year_counts[["year", "topic_label_short", "share"]]

top_k = 6

top_topics = (
    df_temp["topic_label_short"]
    .value_counts()
    .head(top_k)
    .index
)

topic_year_top = topic_year[
    topic_year["topic_label_short"].isin(top_topics)
]

pivot = topic_year_top.pivot(
    index="year",
    columns="topic_label_short",
    values="share"
).fillna(0)

plt.figure(figsize=(10, 5))

for col in pivot.columns:
    plt.plot(
        pivot.index,
        pivot[col],
        marker="o",
        label=col
    )

plt.xlabel("Year")
plt.ylabel("Topic Share")
plt.title("Temporal Changes in Topic Distribution")
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

def ultra_short_label(label):
    if pd.isna(label):
        return "unknown"
    tokens = label.replace("/", ",").split(",")
    main = tokens[0].strip()
    return " ".join(main.split()[:2])


df_temp = df_topics.copy()

df_temp["topic_label_short"] = (
    df_temp["topic_name"]
    .apply(ultra_short_label)
)

# remove outliers if any
df_temp = df_temp[df_temp["topic"] != -1]


topic_year_group = (
    df_temp
    .groupby(["EU_status", "year", "topic_label_short"], as_index=False)
    .size()
)

topic_year_group["share"] = (
    topic_year_group["size"] /
    topic_year_group.groupby(["EU_status", "year"])["size"].transform("sum")
)

top_k = 5

top_topics = (
    df_temp["topic_label_short"]
    .value_counts()
    .head(top_k)
    .index
)

topic_year_group = topic_year_group[
    topic_year_group["topic_label_short"].isin(top_topics)
]


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, group in enumerate(["EU", "Non-EU"]):
    subset = topic_year_group[
        topic_year_group["EU_status"] == group
    ]

    pivot = subset.pivot(
        index="year",
        columns="topic_label_short",
        values="share"
    ).fillna(0)

    for col in pivot.columns:
        axes[i].plot(
            pivot.index,
            pivot[col],
            marker="o",
            label=col
        )

    axes[i].set_title(group)
    axes[i].set_xlabel("Year")

axes[0].set_ylabel("Topic Share")
axes[1].legend(
    title="Topic",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.suptitle("Comparing Temporal Changes in Topics: EU vs Non-EU")
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

def ultra_short_label(label):
    if pd.isna(label):
        return "unknown"
    tokens = label.replace("/", ",").split(",")
    main = tokens[0].strip()
    return " ".join(main.split()[:2])


df_eu = df_topics.copy()

df_eu = df_eu[df_eu["EU_status"] == "EU"]
df_eu = df_eu[df_eu["topic"] != -1]

df_eu["topic_label_short"] = (
    df_eu["topic_name"]
    .apply(ultra_short_label)
)


topic_year_eu = (
    df_eu
    .groupby(["year", "topic_label_short"], as_index=False)
    .size()
)

topic_year_eu["share"] = (
    topic_year_eu["size"] /
    topic_year_eu.groupby("year")["size"].transform("sum")
)

pivot = topic_year_eu.pivot(
    index="topic_label_short",
    columns="year",
    values="share"
).fillna(0)

# keep only 2013 and 2015
pivot = pivot[[2013, 2015]]

pivot["shift_2013_2015"] = pivot[2015] - pivot[2013]

# select biggest absolute shifts
top_shifts = (
    pivot["shift_2013_2015"]
    .abs()
    .sort_values(ascending=False)
    .head(8)
    .index
)

shift_df = pivot.loc[top_shifts].sort_values("shift_2013_2015")


plt.figure(figsize=(8, 5))
plt.barh(
    shift_df.index,
    shift_df["shift_2013_2015"]
)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Change in Topic Share (2015 − 2013)")
plt.title("Biggest Topic Shifts within the EU (2013 → 2015)")
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

def ultra_short_label(label):
    if pd.isna(label):
        return "unknown"
    tokens = label.replace("/", ",").split(",")
    main = tokens[0].strip()
    return " ".join(main.split()[:2])


df_noneu = df_topics.copy()

df_noneu = df_noneu[df_noneu["EU_status"] == "Non-EU"]
df_noneu = df_noneu[df_noneu["topic"] != -1]

df_noneu["topic_label_short"] = (
    df_noneu["topic_name"]
    .apply(ultra_short_label)
)

topic_year_noneu = (
    df_noneu
    .groupby(["year", "topic_label_short"], as_index=False)
    .size()
)

topic_year_noneu["share"] = (
    topic_year_noneu["size"] /
    topic_year_noneu.groupby("year")["size"].transform("sum")
)

pivot = topic_year_noneu.pivot(
    index="topic_label_short",
    columns="year",
    values="share"
).fillna(0)

# keep only 2013 and 2015
pivot = pivot[[2013, 2015]]

pivot["shift_2013_2015"] = pivot[2015] - pivot[2013]

# select biggest absolute shifts
top_shifts = (
    pivot["shift_2013_2015"]
    .abs()
    .sort_values(ascending=False)
    .head(8)
    .index
)

shift_df = pivot.loc[top_shifts].sort_values("shift_2013_2015")


plt.figure(figsize=(8, 5))
plt.barh(
    shift_df.index,
    shift_df["shift_2013_2015"]
)

plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Change in Topic Share (2015 − 2013)")
plt.title("Biggest Topic Shifts within Non-EU (2013 → 2015)")
plt.tight_layout()
plt.show()



