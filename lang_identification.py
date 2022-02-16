import fasttext


class LanguageIdentification:
    def __init__(self):
        pretrained_lang_model = "/tmp/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)
        if len(predictions) > 0:
            return predictions[0][0].replace("__label__", "")
        return "unknown"


if __name__ == "__main__":
    lang_detector = LanguageIdentification()
    print(lang_detector.predict_lang("Questo è un testo italiano!"))
    print(lang_detector.predict_lang("This is an english text!"))
    print(lang_detector.predict_lang("これは日本語のテキストだ"))
