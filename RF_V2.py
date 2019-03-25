import numpy as np


"""
It is assumed that if the number of unique values in a
column is more than 10, it is continuous domain.
Otherwise, it is categorical.
"""


# FILENAME = 'mfdb.csv'
FILENAME = 'test.csv'
MAX_SPLIT_PTS = 50


def entropy():
    """
    Calculate the entropy.
    :return:
    """
    pass


def gini_categorical(data):
    uniq_feat = np.unique(data[:, 0])
    n_class = np.unique(data[:, 1])
    feature_split = np.zeros(shape=(1, len(n_class) + 1))
    for val in uniq_feat:
        count_np = np.array(val)
        for class_ in n_class:
            c = (data[data[:, 0] == val, 1] == class_).sum()
            count_np = np.hstack((count_np, c))
        feature_split = np.vstack((feature_split, count_np))
    feature_split = np.delete(feature_split, 0, axis=0)
    return gini_idx_v1(feature_split)


def gini_continuous(data):
    uniq_feat = np.unique(data[:, 0])
    # generate split points from the uniq_feat.
    # number of split points are assumed to be
    # 2 more than the number of unique features.
    # let number of split points be not more than 50 for now.
    split_pts = list()
    for i in range(0, len(uniq_feat)-1):
        split_pts.append((uniq_feat[i] + uniq_feat[i + 1]) / 2)
    split_pts.insert(0, min(uniq_feat) - 3)
    split_pts.append(max(uniq_feat) + 3)
    split_pts = np.array(split_pts)
    rand_sampl = np.random.choice(split_pts.shape[0],
                                  min(split_pts.shape[0], MAX_SPLIT_PTS),
                                  replace=False)
    # these split points are like the categorical values.
    split_pts = split_pts[rand_sampl]
    n_class = np.unique(data[:, 1])
    feature_split = np.zeros(shape=(1, len(n_class) + 1))
    for val in split_pts:
        count_lt = np.array(val)
        count_gt = np.array(val)
        for class_ in n_class:
            c = (data[data[:, 0] < val, 1] == class_).sum()
            d = (data[data[:, 0] >= val, 1] == class_).sum()
            count_lt = np.hstack((count_lt, c))
            count_gt = np.hstack((count_gt, d))
        feature_split = np.vstack((feature_split, count_lt))
        feature_split = np.vstack((feature_split, count_gt))
    feature_split = np.delete(feature_split, 0, axis=0)
    return gini_idx_v1(feature_split)


def gini_idx_v1(split):
    """
    Calculate the Gini index for the split dataset.
    :return:
    """
    data = split[:, 1:]
    sum_ = np.sum(data, axis=1).reshape((data.shape[0], 1))
    proportion = np.divide(data, sum_)
    gini = 1 - np.sum(np.square(proportion), axis=1)
    gini[np.isnan(gini)] = 1
    return gini


def cross_entropy():
    """
    Calculate the cross entropy.
    :return:
    """
    pass


def min_gini(data):
    # number of unique feature values:
    uniq_feat = np.unique(data[:, 0])
    print('# of Unique values in feature: {}'.format(len(uniq_feat)))
    if len(uniq_feat) < 10:
        gini = gini_categorical(data)
        print(gini)
    else:
        gini = gini_continuous(data)
        print(gini)
    return gini


def split_attr(data_):
    """
    Split the data on attribute.
    :param data_:
    :return:
    """
    class_col = data_.shape[1] - 1
    n_features = data_.shape[1] - 1
    print("# of FEATURES: {}".format(n_features))
    for feat_ in range(n_features):
        print('Feature: {}'.format(feat_))
        # Split the data every way possible (depending on
        # whether the data is continuous or categorical.
        relevant_data = data_[:, [feat_, class_col]]

        # calculate minimum gini
        min_gini(relevant_data)


def main():
    # Get the data:
    # as np array.
    data = np.genfromtxt(FILENAME, delimiter=',', skip_header=True)
    print("ROWS: {}, COLS: {}".format(data.shape[0], data.shape[1]))

    # Split data for every attribute and calculate the minimum Gini
    # index.
    split_attr(data)


if __name__ == '__main__':
    main()
