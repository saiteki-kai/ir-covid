import re
import html
import string
import demoji

import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec


def remove_urls(text):
    regex = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    return re.sub(regex, "", text)


def remove_doi(text):
    regex = r"\b(doi:\s+)?(10[.][0-9]{4,}(?:[.][0-9]+)*(?:(?![\"&\'<>])\S)+)\b"
    return re.sub(regex, "", text)


def remove_punctuation(text):
    punctuation = string.punctuation + "–—‐“”″„’‘•′·«»§¶"
    return "".join([i for i in text if i not in punctuation])


def remove_extra_whitespace(text):
    return " ".join(text.split())


def replace_html_entities(text):
    return html.unescape(text)


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word.lower() not in stop_words])


def remove_emojis(text):
    return demoji.replace(text)


def remove_numbers(text):
    pass


# il lemmatizer di wordnet ha solo questi 4 tag
def pos2wordnet(nltk_pos):
    if nltk_pos.startswith("J"):
        return wn.ADJ
    elif nltk_pos.startswith("V"):
        return wn.VERB
    elif nltk_pos.startswith("N"):
        return wn.NOUN
    elif nltk_pos.startswith("R"):
        return wn.ADV
    else:
        return wn.NOUN


def covid_norm(text, mode="specific"):
    keywords_all = [
        "covid 19",
        "covid-19",
        "sars-cov-2",
        "sarscov2",
        "sars cov 2",
        "2019-nCoV",
        "Wuhan virus",
        "Chinese flu",
        "COronaVIrusDisease",
    ]

    keywords_specific = ["covid 19", "covid-19"]

    if mode == "all":
        keywords = keywords_all
    else:
        keywords = keywords_specific

    for key in keywords:
        text = text.replace(key, "covid19")

    return text


def preprocess(
    text,
    lower=True,
    URLs_remove=True,
    doi_remove=True,
    extra_whitespace_remove=True,
    covid_normalization=False,
    html_remove=True,
    emoji_remove=True,
    punctuation_remove=True,
    stop_words=True,
    lemmatization=True,
):
    sentences = nltk.sent_tokenize(text)

    tokens_list = []
    for sent in sentences:
        if lower:
            sent = sent.lower()
        if URLs_remove:
            sent = remove_urls(sent)
        if doi_remove:
            sent = remove_doi(sent)
        if extra_whitespace_remove:
            sent = remove_extra_whitespace(sent)
        if covid_normalization:
            # Since it's a collection of scientific
            # papers we think it's better not to
            # normalize the word by default.
            sent = covid_norm(sent)
        if html_remove:
            sent = replace_html_entities(sent)  # before removing punctuation (&\w+;)
        if emoji_remove:
            sent = remove_emojis(sent)
        if punctuation_remove:
            sent = remove_punctuation(sent)
        if stop_words:
            sent = remove_stopwords(sent)

        tokens = nltk.word_tokenize(sent)
        tokens_list.append(tokens)

    tagged_tokens = nltk.pos_tag_sents(tokens_list)

    wnl = WordNetLemmatizer()

    result = []
    for tokens in tagged_tokens:
        for (token, pos) in tokens:
            if token == "covid" and covid_normalization:
                token = "covid19"

            if lemmatization:
                token = wnl.lemmatize(token, pos2wordnet(pos))

            result.append(token)
    return result

# Most similar word with Word2Vec

def most_similar(word:str, wv_model=None):
    if wv_model is None:
        wv_model = Word2Vec.load("data/word2vec.model")

    try:
        most_similar_word = wv_model.wv.most_similar(word, topn=1)
        most_similar_word = most_similar_word[0][0]
    except:
        most_similar_word = None
    
    return most_similar_word


def query_similar_words(query:str, wv_model=None):
    new_query = ""
    query_tk = [t.split() for t in  nltk.sent_tokenize(query)][0]
    if wv_model is None:
        wv_model = Word2Vec.load("data/word2vec.model")
    for word in query_tk:
        new_query += " " + word
        ms = most_similar(word, wv_model)
        if ms is not None:
            new_query += " " + ms

    new_query = new_query[1:] # Just delete the first free space
    return new_query

def reduce_queries(queries, narrative_threshold=10, description_threshold=10):
    
    narrative = queries.narrative.str.split(expand=True).stack().value_counts().to_dict()
    description = queries.description.str.split(expand=True).stack().value_counts().to_dict()

    to_remove_word = []

    for key in narrative:
        if narrative[key] > narrative_threshold:
            to_remove_word.append(key)

    for key in description:
        if description[key] > description_threshold:
            to_remove_word.append(key)


    query = queries.copy()
    for index, row in query.iterrows():
        description = [t.split() for t in  nltk.sent_tokenize(row["description"])][0]
        narrative = [t.split() for t in  nltk.sent_tokenize(row["narrative"])][0]

        new_description = ""
        for word in description:
            if word in to_remove_word:
                new_description += ""
            else:
                new_description += " " + word
        new_description = new_description[0:]

        new_narrative = ""
        for word in narrative:
            if word in to_remove_word:
                new_narrative += ""
            else:
                new_narrative += " " + word
        new_narrative = new_narrative[0:]

        row["description"] = new_description
        row["narrative"] = new_narrative

    return query   