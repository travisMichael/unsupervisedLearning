import sys
from cluster.kmeans.cluster_plot import generate_cardio_elbow_plot, generate_loan_elbow_plot


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please specify filename and data set to pre-process")
    else:
        plot = sys.argv[1]
        if plot == 'elbow':
            generate_cardio_elbow_plot('cluster/kmeans/')
            generate_loan_elbow_plot('cluster/kmeans/')
