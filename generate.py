import sys
from cluster.kmeans.cluster_plot import generate_cardio_elbow_plot, generate_loan_elbow_plot
from decomposition.pca import pca_cardio_scatter_plot, pca_loan_scatter_plot
from decomposition.ica import ica_cardio_scatter_plot, ica_loan_scatter_plot
from decomposition.random_projection import grp_cardio_scatter_plot, grp_loan_scatter_plot
from decomposition.svd import fa_cardio_scatter_plot, fa_loan_scatter_plot


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please specify filename and data set to pre-process")
    else:
        plot = sys.argv[1]
        if plot == 'elbow':
            generate_cardio_elbow_plot('cluster/kmeans/')
            generate_loan_elbow_plot('cluster/kmeans/')
        elif plot == 'decomposition':
            pca_cardio_scatter_plot('')
            pca_loan_scatter_plot('')
            ica_cardio_scatter_plot('')
            ica_loan_scatter_plot('')
            grp_cardio_scatter_plot('')
            grp_loan_scatter_plot('')
            fa_cardio_scatter_plot('')
            fa_loan_scatter_plot('')
