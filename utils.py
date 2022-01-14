import re
import html
import string
import demoji

import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer


def remove_urls(text):
	regex = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
	return re.sub(regex, "", text)


def remove_doi(text):
	regex = r"\b(doi:\s+)?(10[.][0-9]{4,}(?:[.][0-9]+)*(?:(?![\"&\'<>])\S)+)\b"
	return re.sub(regex, "", text)


# decidere se tenere qualche carattere
def remove_punctuation(text):
	punctuation = string.punctuation + "“”"
	# punctuation = punctuation.replace("+", "")
    # punctuation = punctuation.replace("-", "")
    # punctuation = punctuation.replace('"', "")
    # punctuation = punctuation.replace("{", "")
    # punctuation = punctuation.replace("}", "")
    # punctuation = punctuation.replace("(", "")
    # punctuation = punctuation.replace(")", "")
    # punctuation = punctuation.replace(".", "")
    # punctuation = punctuation.replace(":", "")
    # punctuation = punctuation.replace("~", "")
    # punctuation = punctuation.replace("^", "")
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
	lemmatization=True
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
