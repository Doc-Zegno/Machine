

class TagStats:
    def __init__(self):
        self.num_gold = 0
        self.num_predicted = 0
        self.num_gold_predicted = 0


class TagScore:
    def __init__(self, tags):
        self.stats = {tag: TagStats() for tag in tags}


class ChunkScore:
    def __init__(self, chunks):
        self.stats = {chunk: [0, 0] for chunk in chunks}
        self.num_retrieved = 0
        self.num_relevant = 0
        self.num_retrieved_relevant = 0


class ParserScore:
    def __init__(self):
        self.num_unlabeled_arcs = 0
        self.num_labeled_arcs = 0
        self.num_modified_unlabeled_arcs = 0
        self.num_unlabeled_trees = 0
        self.num_labeled_trees = 0
        self.num_modified_unlabeled_trees = 0
        self.num_labels = 0
