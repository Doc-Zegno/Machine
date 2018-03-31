import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError("too few arguments: path to report expected")

    f1_series = []
    total_words = 0

    for line in open(sys.argv[1]):
        line = line.split()
        if len(line) != 12:
            if len(line) == 3:
                if line[0] == "Total" and line[1] == "words:":
                    total_words = int(line[2])
            continue

        if line[-5] != "F1":
            continue

        f1 = float(line[-3][:-1])
        num_words = int(line[-2][1:])
        f1_series.append((f1, num_words))

    total_f1 = 0.0
    for f1, num_words in f1_series:
        total_f1 += f1 * num_words / total_words

    print("Weighted F1: {:4.2f}%".format(total_f1))
