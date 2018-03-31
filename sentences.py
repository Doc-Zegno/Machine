import nltk
import nltk.corpus as corpus


# Transforms sentences from corpus.
# Returns sentences with lowercase words
class LowercaseSentences:
    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        """Return the next sentence from corpus with lowercase words."""
        for sent in self.sents:
            lowered_sent = []
            for word in sent:
                lowered_sent.append(word.lower())
            yield lowered_sent


# Transforms sentences from corpus.
# Returns tags only
class TagsFromSentences:
    def __init__(self, sents):
        self.sents = sents

    def __iter__(self):
        """Return the next sentence from corpus with lowercase words."""
        for sent in self.sents:
            tags = []
            for word, tag in sent:
                tags.append(tag)
            yield tags


# Iterator factories.
# Lowercase sentences iterator factory
class LowercaseSentencesFactory:
    def __call__(self, sents):
        return LowercaseSentences(sents)


if __name__ == "__main__":
    sents = corpus.conll2000.tagged_sents()[:10]
    for tags in TagsFromSentences(sents):
        print(tags)
