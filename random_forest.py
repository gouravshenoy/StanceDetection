from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self, train_file, test_file):
        self.__train_file = train_file
        self.__test_file = test_file
        self.__train_data = []
        self.__train_labels = []
        self.__test_data = []
        self.__test_labels = []

    def read_data(self):
        with open(self.__train_file, 'rU') as rp:
            for row in rp:
                train_data = row.split(',')
                train_label = train_data[train_data.__len__() - 1]
                train_data.remove(train_label)
                # print ("train data = {}, \n\n train_label = {}".format(train_data, train_label))

                self.__train_data.append(train_data)
                self.__train_labels.append(train_label)

            print ("num. train examples = {}, num. train labels = {}".format(self.__train_data.__len__(),
                                                                             self.__train_labels.__len__()))

        with open(self.__test_file, 'rU') as rp:
            for row in rp:
                test_data = row.split(',')
                test_label = test_data[test_data.__len__() - 1]
                test_data.remove(test_label)
                # print ("test data = {}, \n\n test_label = {}".format(test_data, test_label))

                self.__test_data.append(test_data)
                self.__test_labels.append(test_label)

            print ("num. test examples = {}, num. test labels = {}".format(self.__test_data.__len__(),
                                                                             self.__test_labels.__len__()))


    def learn(self):
        forest = RandomForestClassifier(n_estimators=100)
        forest = forest.fit(self.__train_data, self.__train_labels)

        self.__predicted_labels = forest.predict(self.__test_data)


    def calculate_accuracy(self):

        misclassification_count = 0
        for index, predicted_label in enumerate(self.__predicted_labels):
            if(not self.__test_labels[index] == predicted_label):
                misclassification_count += 1

        accuracy = float(self.__test_labels.__len__() - misclassification_count) / float(self.__test_labels.__len__())

        print ("Random Forest Accuracy = {}".format(accuracy))
        pass

if __name__ == "__main__":
    random_forest = RandomForest("../../Files/bow_feature_vectors/3_pos_tags/train/donald_train_bow_features.csv",
                                 "../../Files/bow_feature_vectors/3_pos_tags/test/donald_test_bow_features.csv")

    random_forest.read_data()
    random_forest.learn()
    random_forest.calculate_accuracy()