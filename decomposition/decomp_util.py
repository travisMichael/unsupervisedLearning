from sklearn.metrics import mean_squared_error
from time import time


def reconstruct(estimator, f, train):
    print('..')
    pca_data = estimator.fit_transform(train)

    reconstructed_data = estimator.inverse_transform(pca_data)

    error = mean_squared_error(train, reconstructed_data)

    f.write("%.3f\t%.3f\t%.3f\n" % (1, error, 0.0))


def time_estimator(estimator, f, train):
    print('..')
    t = 0.0
    a = 100
    for i in range(a):
        start = time()
        estimator.fit_transform(train)
        t += time() - start
    f.write("%.3f\t%.3f\t%.3f\n" % (1, t/a , 0.0))
