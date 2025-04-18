import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st


nlp = spacy.load("en_core_web_sm")

@st.cache_data(show_spinner=True)
def scrape_geo_article_titles():
    url = "https://www.geo.tv/latest-news"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    articles_data = []
    for h2 in soup.select("h2[data-vr-headline]"):
        title = h2.get_text(strip=True)
        link_tag = h2.find_parent("a")
        href = link_tag["href"] if link_tag and "href" in link_tag.attrs else None
        if title and href:
            full_url = "https://www.geo.tv" + href if href.startswith("/") else href
            articles_data.append({"title": title, "url": full_url})

    return pd.DataFrame(articles_data)

def extract_named_entities_from_titles(df):
    all_text = " ".join(df["title"].dropna())
    doc = nlp(all_text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    places = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    things = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT"]]
    return {"names": Counter(names), "places": Counter(places), "things": Counter(things)}

def plot_top_entities(entity_counter, title, top_n=10):
    most_common = entity_counter.most_common(top_n)
    if not most_common:
        st.warning(f"No data found for {title}")
        return
    labels, values = zip(*most_common)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(values), y=list(labels), ax=ax)
    ax.set_title(f"Top {top_n} {title}")
    st.pyplot(fig)

def plot_wordcloud(entity_counter, title):
    if not entity_counter:
        st.warning(f"No data to generate word cloud for {title}")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(entity_counter)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f"{title} Word Cloud")
    st.pyplot(fig)

st.title("📰 Geo.TV News Headline Analyzer")
st.write("Scrapes latest headlines from Geo.TV and extracts names, places, and organizations using spaCy.")

if st.button("Scrape & Analyze"):
    with st.spinner("Scraping Geo.TV..."):
        df_articles = scrape_geo_article_titles()

    if df_articles.empty:
        st.error("No headlines scraped.")
    else:
        st.success(f"✅ Scraped {len(df_articles)} headlines!")
        st.dataframe(df_articles.head())

        entities = extract_named_entities_from_titles(df_articles)

        st.subheader("📈 Top Entities")
        plot_top_entities(entities["names"], "Names")
        plot_top_entities(entities["places"], "Places")
        plot_top_entities(entities["things"], "Things")

        st.subheader("🌥️ Word Clouds")
        plot_wordcloud(entities["names"], "Names")
        plot_wordcloud(entities["places"], "Places")
        plot_wordcloud(entities["things"], "Things")
