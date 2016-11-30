import csv
import nltk
from nltk import word_tokenize


class POS:
    nouns = []
    adjs = []
    verbs = []
    features_3_pos = []
    features_all_words = []

    def extract_bow_3_pos_tags(self, filename):
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

        self.features_3_pos.extend(self.nouns)
        self.features_3_pos.extend(self.verbs)
        self.features_3_pos.extend(self.adjs)

        # remove duplicate features
        self.features_3_pos = list(set(self.features_3_pos))
        print("features (after duplicates removed) are {}".format(self.features_3_pos.__len__()))


    def extract_bow_all_words(self, filename):
        """
        2. a) Extract a bag-of-words list of all words for all targets individually
        :param filename:
        :return:
        """
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                # print(row)
                self.features_all_words.append(row[0])

        # remove duplicate features
        self.features_all_words = list(set(self.features_all_words))
        print("features (after duplicates removed) are {}".format(self.features_all_words.__len__()))

    def create_file(self, filename, new_filename, all_words=False):
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
            # if not all_words:
            #     writer.writerow(self.features_3_pos)
            # else:
            #     writer.writerow(self.features_all_words)

            for row in reader:
                r = []
                text = word_tokenize(row[0])

                if not all_words:
                    for feature in self.features_3_pos:
                        if feature in text:
                            r.append('1')
                        else:
                            r.append('0')
                else:
                    for feature in self.features_all_words:
                        if feature in text:
                            r.append('1')
                        else:
                            r.append('0')

                r.append(row[2]) # stance column
                writer.writerow(r)


if __name__ == '__main__':
    pos = POS()

    # pos.extract_bow_3_pos_tags("../../Files/pos_tagged/train/legalization_train_pos_tagged.txt")
    # pos.extract_bow_3_pos_tags("../../Files/pos_tagged/test/legalization_test_pos_tagged.txt")
    # pos.create_file("../../Files/train_test_files/train/legalization_train.csv",
    #                 "../../Files/bow_feature_vectors/3_pos_tags/train/legalization_train_bow_features.csv")
    # pos.create_file("../../Files/train_test_files/test/legalization_test.csv",
    #                 "../../Files/bow_feature_vectors/3_pos_tags/test/legalization_test_bow_features.csv")

    pos.extract_bow_all_words("../../Files/pos_tagged/train/legalization_train_pos_tagged.txt")
    pos.extract_bow_all_words("../../Files/pos_tagged/test/legalization_test_pos_tagged.txt")

    pos.create_file("../../Files/train_test_files/train/legalization_train.csv",
                    "../../Files/bow_feature_vectors/all_words/train/legalization_train_bow_all_features.csv",
                    all_words=True)
    pos.create_file("../../Files/train_test_files/test/legalization_test.csv",
                    "../../Files/bow_feature_vectors/all_words/test/legalization_test_bow_all_features.csv",
                    all_words=True)