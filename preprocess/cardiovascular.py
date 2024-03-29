import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, minmax_scale
from sklearn.model_selection import train_test_split

# def normalize_gender_attribute(x):
#     if x == 1:
#         return 0.0
#     else:
#         return 1.0


def pre_process_cardio_data(path):
    df = pd.read_csv(path + "cardio_train.csv", delimiter=";")
    print(df.shape)

    # max_stats = df.max()
    # min_stats = df.min()

    # df_positive = df[df['cardio'] == 1]
    # df_negative = df[df['cardio'] == 0]
    # df = df_negative

    # df['age'] = df['age'].apply(lambda x: x / max_stats['age'])
    # df['height'] = df['height'].apply(lambda x: x / max_stats['height'])
    # df['weight'] = df['weight'].apply(lambda x: x / max_stats['weight'])
    # df['ap_hi'] = df['ap_hi'].apply(lambda x: x / max_stats['ap_hi'])
    # df['ap_lo'] = df['ap_lo'].apply(lambda x: x / max_stats['ap_lo'])
    # df['gender'] = df['gender'].apply(lambda x: normalize_gender_attribute(x))

    y = np.array(df.iloc[:,12])
    x = df.iloc[:,1:12]

    X_train, X_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=44)

    rob_scaler = StandardScaler()

    X_train_R = rob_scaler.fit_transform(X_train)
    X_test_R = rob_scaler.transform(X_test)

    negative_cases = len(df[(df.cardio == 0)])
    positive_cases = len(df[(df.cardio == 1)])
    print('Cases', positive_cases, negative_cases)

    # x_train = x.iloc[0:56000, :].values
    # y_train = y[0:56000]
    #
    # x_test = x.iloc[56000:, :].values
    # y_test = y[56000:]

    print(X_train_R.shape)
    print(y_train.shape)

    print(X_test_R.shape)
    print(y_test.shape)

    if not os.path.exists(path + 'data/cardio/train'):
        os.makedirs(path + 'data/cardio/train')
    if not os.path.exists(path + 'data/cardio/test'):
        os.makedirs(path + 'data/cardio/test')

    x_train_file = open(path + 'data/cardio/train/x', 'wb')
    y_train_file = open(path + 'data/cardio/train/y', 'wb')
    x_test_file = open(path + 'data/cardio/test/x', 'wb')
    y_test_file = open(path + 'data/cardio/test/y', 'wb')

    pickle.dump(X_train_R, x_train_file)
    pickle.dump(y_train, y_train_file)
    pickle.dump(X_test_R, x_test_file)
    pickle.dump(y_test, y_test_file)

    x_train_file.close()
    y_train_file.close()
    x_test_file.close()
    y_test_file.close()

    print("done")


if __name__ == "__main__":
    pre_process_cardio_data('../')
