import sys
from cluster.kmeans.cluster_plot import generate_cardio_elbow_plot, generate_loan_elbow_plot
from decomposition.pca import pca_cardio_scatter_plot, pca_loan_scatter_plot
from decomposition.ica import ica_cardio_scatter_plot, ica_loan_scatter_plot
from decomposition.random_projection import grp_cardio_scatter_plot, grp_loan_scatter_plot
from decomposition.svd import fa_cardio_scatter_plot, fa_loan_scatter_plot
from cluster.kmeans.k_means_cardio_visualizations import run_k_means_on_cardio_data, run_k_means_on_loan_data, run_cardio_benchmark, run_loan_benchmark
from cluster.expectation_maximization.homog_score import calculate_cardio_EM_homogeneity_scores, calculate_cardio_kmeans_homogeneity_scores, calculate_loan_EM_homogeneity_scores, calculate_loan_kmeans_homogeneity_scores
from decomposition.pca import generate_loan_pca_reconstruction_stats, generate_cardio_pca_reconstruction_stats, generate_pca_variance
from decomposition.pca import pca_runtime_stats
from decomposition.ica import ica_runtime_stats
from decomposition.svd import svd_runtime_stats
from decomposition.random_projection import grp_runtime_stats
from decomposition.pca_distribution import pca_cardio_distribution_plot, pca_loan_distribution_plot
from decomposition.ica_distribution import ica_cardio_distribution_plot, ica_loan_distribution_plot
from decomposition.grp_distribution import grp_cardio_distribution_plot, grp_loan_distribution_plot
from supervised.nn_stats import generate_neural_net_runtime_stats
from supervised.nn_cluster_stats import generate_decomp_neural_net_runtime_stats

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
            calculate_loan_kmeans_homogeneity_scores('')
            calculate_loan_EM_homogeneity_scores('')
            calculate_cardio_kmeans_homogeneity_scores('')
            calculate_cardio_EM_homogeneity_scores('')
        # reconstruction
        elif plot == 'table_5':
            generate_loan_pca_reconstruction_stats('')
            generate_cardio_pca_reconstruction_stats('')
        # kurtosis values
        elif plot == 'table_6':
            pca_cardio_distribution_plot('')
            pca_loan_distribution_plot('')
            ica_cardio_distribution_plot('')
            ica_loan_distribution_plot('')
            grp_loan_distribution_plot('')
            grp_cardio_distribution_plot('')
        # pca variance
        elif plot == 'figure_3':
            generate_pca_variance('', 'loan')
            generate_pca_variance('', 'cardio')
        # distributions
        elif plot == 'figure_4_and_5':
            pca_cardio_distribution_plot('')
            pca_loan_distribution_plot('')
            ica_cardio_distribution_plot('')
            ica_loan_distribution_plot('')
            grp_loan_distribution_plot('')
            grp_cardio_distribution_plot('')
        # nn_runtime metrics
        elif plot == 'table_7':
            generate_neural_net_runtime_stats('')
        # decomp runtimes
        elif plot == 'figure_6':
            pca_runtime_stats('', 'loan')
            ica_runtime_stats('', 'loan')
            svd_runtime_stats('', 'loan')
            grp_runtime_stats('', 'loan')
        # nn accuracy scores
        elif plot == 'table_8':
            generate_decomp_neural_net_runtime_stats('')
