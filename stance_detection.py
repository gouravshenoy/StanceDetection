import csv
import nltk
from nltk import word_tokenize


class POS:
    nouns = []
    adjs = []
    verbs = []
    features = []

    def categorise(self, filename):
        """
        2. a) Extract a bag-of-words list of nouns, adjectives, and verbs for all targets individually
        :param filename:
        :return:
        """
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                # print(row)
                if row[-1] in ['NN', 'NNS', 'NNP']:
                    self.nouns.append(row[0])
                elif row[-1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    self.verbs.append(row[0])
                elif row[-1] in ['JJ', 'JJR', 'JJS']:
                    self.adjs.append(row[0])

        self.features.extend(self.nouns)
        self.features.extend(self.verbs)
        self.features.extend(self.adjs)
        print("features are {}".format(self.features.__len__()))

    def create_file(self, filename, new_filename):
        """
        Use those words as features and create a file in which the feature values are either 1 or 0 depending
        on whether the corresponding word is in the tweet or not. Add the tweet label as the last element
        (gold class) in every line.
        :param filename:
        :return:
        """
        with open(new_filename, 'w') as wp, open(filename, 'rU') as rp:
            writer = csv.writer(wp, delimiter = ',')
            reader = csv.reader(rp, delimiter = ',')
            writer.writerow(self.features)
            for row in reader:
                r = []
                text = word_tokenize(row[0])
                for feature in self.features:
                    if feature in text:
                        r.append('1')
                    else:
                        r.append('0')
                r.append(row[2]) # stance column
                writer.writerow(r)


if __name__ == '__main__':
    pos = POS()
    pos.categorise("legalization_train.txt")
    pos.categorise("legalization_test.txt")
    pos.create_file("legalization_train.csv", "parse_legalization_train.csv")
    pos.create_file("legalization_test.csv", "parse_legalization_test.csv")