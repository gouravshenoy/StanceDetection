import csv
import operator
from nltk import word_tokenize
from nltk import stem


class SubLexicon:
    def __init__(self, word, pos, isStem, polarity):
        self.__word = word
        self.__pos = self.get_pos(pos)
        self.__stemmed = \
            True if isStem == 'y' else False  # True if stemmed=y, else False
        self.__polarity = polarity

    def get_pos(self, pos):
        if pos == 'noun':
            return ['NN', 'NNS', 'NNP', 'NNPS']
        elif pos == 'adj':
            return ['JJ', 'JJR', 'JJS']
        elif pos == 'verb':
            return ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        else:
            return ['ANY']

    def get_polarity(self):
        return self.__polarity

class POS:
    nouns = []
    adjs = []
    verbs = []
    features_3_pos = []
    features_all_words = []
    subjectivity_lexicons = {}

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
            writer = csv.writer(wp, delimiter=',')
            reader = csv.reader(rp, delimiter=',')
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

                r.append(row[2])  # stance column
                writer.writerow(r)

    def read_subjectivity_lexicons(self, sublex_filename):
        """
        Reads the subjectivity lexicon file, and constructs the datastructure,
        finally adds to the 'subjectivity_lexicons' list
        :param sublex_filename:
        :return:
        """
        with open(sublex_filename) as rp:
            for row in rp:
                line_words = word_tokenize(row)
                # print line_words

                # parse the line
                lexicon_word = line_words[2].split('=')[1]
                lexicon_pos = line_words[3].split('=')[1]
                lexicon_stemmed = line_words[4].split('=')[1]
                lexicon_polarity = line_words[5].split('=')[1]

                # create new DS object & add to list
                self.subjectivity_lexicons[lexicon_word] = SubLexicon(word=lexicon_word,
                                                                      pos=lexicon_pos,
                                                                      isStem=lexicon_stemmed,
                                                                      polarity=lexicon_polarity)

        print ("total num. subjectivity lexicons = {}".format(self.subjectivity_lexicons.__len__()))
        print ("test lexicon polarity = {}".format(self.subjectivity_lexicons['abandoned'].get_polarity()))
        pass

    def create_features_with_sublex(self, filename, new_filename):
        with open(new_filename, 'w') as wp, open(filename, 'rU') as rp:
            writer = csv.writer(wp, delimiter=',')
            reader = csv.reader(rp, delimiter=',')

            # print (self.features_all_words)

            # create stemmer for extracting stems of words
            stemmer = stem.PorterStemmer()
            for row in reader:
                r = []
                text = word_tokenize(row[0])

                for feature in self.features_all_words:
                    if feature in text:
                        # check if feature or the stem of the feature
                        #   is in subjectivity lexicon
                        feature_stem = stemmer.stem(feature)
                        feature = feature.lower()
                        if feature_stem in self.subjectivity_lexicons:
                            lexicon_obj = self.subjectivity_lexicons[feature_stem]
                            if lexicon_obj.get_polarity() == 'positive':
                                r.append('1')
                            else:
                                r.append('-1')
                        elif feature in self.subjectivity_lexicons:
                            lexicon_obj = self.subjectivity_lexicons[feature]
                            if lexicon_obj.get_polarity() == 'positive':
                                r.append('1')
                            else:
                                r.append('-1')
                        else:
                            r.append('0')
                    else:
                        r.append('0')

                r.append(row[2])  # stance column
                writer.writerow(r)

        pass

    def calculate_baseline(self, train_filename, test_filename):
        with open(train_filename, 'rU') as rp, open(test_filename, 'rU') as tp:
            reader_train = csv.reader(rp, delimiter=',')
            reader_test = csv.reader(tp, delimiter=',')

            # num. of examples in test
            test_data = list(reader_test)
            total_test_size = len(test_data)

            # count number of classes
            classes = {'FAVOR': 0, 'AGAINST': 0, 'NONE': 0}
            for row in reader_train:
                classes[row[2]] += 1

            max_class = max(classes.iteritems(), key=operator.itemgetter(1))[0]
            print ("max class for this data-set is: {}".format(max_class))

            misclassification_count = 0
            for row in test_data:
                if not row[2] == max_class:
                    misclassification_count += 1

            baseline_accuracy = float(total_test_size - misclassification_count) / float(total_test_size)
            print ("data-set size = {}, misclassification count = {}. Hence baseline accuracy = {}".format(total_test_size, misclassification_count, baseline_accuracy))
        pass

if __name__ == '__main__':
    pos = POS()

    # pos.extract_bow_3_pos_tags("../../Files/pos_tagged/train/feminist_train_pos_tagged.txt")
    # pos.extract_bow_3_pos_tags("../../Files/pos_tagged/test/feminist_test_pos_tagged.txt")
    #
    # pos.create_file("../../Files/train_test_files/train/feminist_train.csv",
    #                 "../../Files/bow_feature_vectors/3_pos_tags/train/feminist_train_bow_features.csv")
    # pos.create_file("../../Files/train_test_files/test/feminist_test.csv",
    #                 "../../Files/bow_feature_vectors/3_pos_tags/test/feminist_test_bow_features.csv")

    # pos.extract_bow_all_words("../../Files/pos_tagged/train/hillary_train_pos_tagged.txt")
    # pos.extract_bow_all_words("../../Files/pos_tagged/test/donald_test_pos_tagged.txt")
    #
    # pos.create_file("../../Files/train_test_files/train/hillary_train.csv",
    #                 "../../Files/bow_feature_vectors/all_words/train/donald_train_bow_all_features.csv",
    #                 all_words=True)
    # pos.create_file("../../Files/train_test_files/test/donald_test.csv",
    #                 "../../Files/bow_feature_vectors/all_words/test/donald_test_bow_all_features.csv",
    #                 all_words=True)

    # pos.read_subjectivity_lexicons('../../Files/lexicons/subjectivity/subjclueslen1-HLTEMNLP05.tff')
    # pos.create_features_with_sublex("../../Files/train_test_files/train/feminist_train.csv",
    #                                 "../../Files/bow_feature_vectors/sublex_all_words/train/feminist_train_sublex_all_features.csv")
    # pos.create_features_with_sublex("../../Files/train_test_files/test/feminist_test.csv",
    #                                 "../../Files/bow_feature_vectors/sublex_all_words/test/feminist_test_sublex_all_features.csv")

    pos.calculate_baseline("../../Files/train_test_files/train/hillary_train.csv",
                           "../../Files/train_test_files/test/donald_test.csv")