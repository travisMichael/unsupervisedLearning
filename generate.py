import sys
from cluster.kmeans.cluster_plot import generate_cardio_elbow_plot, generate_loan_elbow_plot
from decomposition.pca import pca_cardio_scatter_plot, pca_loan_scatter_plot
from decomposition.ica import ica_cardio_scatter_plot, ica_loan_scatter_plot
from decomposition.random_projection import grp_cardio_scatter_plot, grp_loan_scatter_plot
from decomposition.svd import fa_cardio_scatter_plot, fa_loan_scatter_plot
from cluster.kmeans.k_means_cardio_visualizations import run_k_means_on_cardio_data, run_k_means_on_loan_data, run_cardio_benchmark, run_loan_benchmark


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Please specify filename and data set to pre-process")
    else:
        plot = sys.argv[1]
        if plot == 'figure_1_and_2':
            run_cardio_benchmark('')
            run_cardio_benchmark('')
            run_k_means_on_cardio_data('')
            run_k_means_on_loan_data('')
            generate_cardio_elbow_plot('cluster/kmeans/')
            generate_loan_elbow_plot('cluster/kmeans/')
        # homogeneity scores
        elif plot == 'table_1_and_2_and_3_and_4':
            pass
        # reconstruction
        elif plot == 'table_5':
            pass
        # kurtosis values
        elif plot == 'table_6':
            pass
        # pca variance
        elif plot == 'figure_3':
            pass
        # distributions
        elif plot == 'figure_4_and_5':
            pass
        # nn_runtime metrics
        elif plot == 'table_7':
            pass
        # decomp runtimes
        elif plot == 'figure_6':
            pass
        # nn accuracy scores
        elif plot == 'table_8':
            pass
