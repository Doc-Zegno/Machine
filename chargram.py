import nltk
import nltk.corpus as corpus


BEGIN_CHAR = '%'  # '%'
END_CHAR = ''    # '#'
MIN_LEN = 1      # 2
MAX_LEN = 4      # 5


# Process a batch of sentences
class CharGramSentences:
    def __init__(self, sents, begin_char=BEGIN_CHAR, end_char=END_CHAR,
                 min_len=MIN_LEN, max_len=MAX_LEN):
        self.sents = sents
        self.grammer = CharGram(begin_char, end_char, min_len, max_len)

    def __iter__(self):
        """Taking a sequence of sentences return sequence of character ngrams."""
        # Separately process each sentence and form a new one
        for sent in self.sents:
            new_sent = []
            for word in sent:
                new_sent.extend(self.grammer(word))
            yield new_sent


# Process one word
class CharGram:
    def __init__(self, begin_char=BEGIN_CHAR, end_char=END_CHAR,
                 min_len=MIN_LEN, max_len=MAX_LEN):
        self.begin_char = begin_char
        self.end_char = end_char
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, word):
        """Taking one word as input return the list of its character ngrams."""
        # Append begin and end characters
        word = word.lower()
        word = "{}{}{}".format(self.begin_char, word, self.end_char)
        grams = []

        # Create and store all ngrams for n from min_len to max_len
        # for cur_len in range(self.min_len, self.max_len + 1):
        #     for offset in range(0, len(word) - cur_len + 1):
        #         grams.append(word[offset:offset + cur_len])
        # return grams

        max_gram_len = min(self.max_len, len(word))
        # for cur_len in range(self.min_len, max_gram_len + 1):
        #     grams.append(word[:cur_len])
        for cur_len in range(self.min_len, max_gram_len + 1):
            grams.append(word[-cur_len:])
        return grams


if __name__ == "__main__":
    TEST_RANGE = 10
    sents = corpus.brown.sents()[:TEST_RANGE]
    print("sents:")

    total_len = 0
    for sent in sents:
        total_len += len(sent)
    print("len:", total_len)

    grams = [sent for sent in CharGramSentences(sents, min_len=MIN_LEN, max_len=MAX_LEN)]
    print("\ngrams:")
    print(grams)

    total_len = 0
    for sent in grams:
        total_len += len(sent)
    print("len:", total_len)
