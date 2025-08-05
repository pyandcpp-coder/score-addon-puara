#include "TSNEAvnd.hpp"
#include <Eigen/Dense>
#include <random>
#include <cmath>

namespace puara_gestures::objects
{

void TSNEAvnd::operator()()
{
  if (!inputs.reset.value.has_value()) {
    m_is_computed = false;
    m_current_iteration = 0;
    m_embedded_data.clear();
  }

  const auto& input_vec = inputs.data.value;
  const int n_features = inputs.n_features.value;
  const int output_dims = inputs.output_dims.value;
  const double perplexity = inputs.perplexity.value;
  const double learning_rate = inputs.learning_rate.value;
  const int max_iterations = inputs.max_iterations.value;

  if (input_vec.empty() || n_features <= 0 || (input_vec.size() % n_features != 0)) {
    return;
  }

  const int n_samples = input_vec.size() / n_features;
  if (n_samples < 2) {
    return;
  }
  if (m_cached_n_samples != n_samples || m_cached_n_features != n_features ||
     m_cached_output_dims != output_dims) {
    m_data_matrix.resize(n_samples, n_features);
    m_distances.resize(n_samples, n_samples);
    m_probabilities_p.resize(n_samples, n_samples);
    m_probabilities_q.resize(n_samples, n_samples);
    m_embedding.resize(n_samples, output_dims);
    m_gradient.resize(n_samples, output_dims);
    m_velocity.resize(n_samples, output_dims);
    m_beta.resize(n_samples);
    m_entropy.resize(n_samples);

    m_cached_n_samples = n_samples;
    m_cached_n_features = n_features;
    m_cached_output_dims = output_dims;
    m_is_computed = false;
    m_current_iteration = 0;
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      data_view(input_vec.data(), n_samples, n_features);
  m_data_matrix = data_view;

  if (!m_is_computed) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1e-4);

    for (int i = 0; i < n_samples; ++i) {
      for (int j = 0; j < output_dims; ++j) {
        m_embedding(i, j) = dist(gen);
      }
    }

    m_velocity.setZero();
    computeGaussianKernel();
    m_is_computed = true;
  }

  if (m_current_iteration < max_iterations) {
    computeQDistribution();
    computeGradient();

    // Update embedding with momentum
    const double momentum = (m_current_iteration < 250) ? 0.5 : 0.8;
    m_velocity = momentum * m_velocity - learning_rate * m_gradient;
    m_embedding += m_velocity;

    // Center embedding
    m_embedding.rowwise() -= m_embedding.colwise().mean();

    m_current_iteration++;
  }

  // Output results
  m_embedded_data.assign(m_embedding.data(), m_embedding.data() + m_embedding.size());
  outputs.embedded_data.value = m_embedded_data;
  outputs.kl_divergence.value = computeKLDivergence();
}

void TSNEAvnd::computeGaussianKernel()
{
  const int n_samples = m_cached_n_samples;
  const double target_entropy = std::log(inputs.perplexity.value);

  // Compute pairwise distances
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_samples; ++j) {
      if (i != j) {
        m_distances(i, j) = (m_data_matrix.row(i) - m_data_matrix.row(j)).squaredNorm();
      } else {
        m_distances(i, j) = 0.0;
      }
    }
  }
  for (int i = 0; i < n_samples; ++i) {
    double beta_min = 0.0;
    double beta_max = std::numeric_limits<double>::max();
    double beta = 1.0;

    for (int iter = 0; iter < 50; ++iter) {
      double sum_p = 0.0;
      double entropy = 0.0;

      for (int j = 0; j < n_samples; ++j) {
        if (i != j) {
          double p_ij = std::exp(-beta * m_distances(i, j));
          sum_p += p_ij;
          entropy += beta * m_distances(i, j) * p_ij;
        }
      }

      if (sum_p > 0) {
        entropy = (entropy / sum_p) + std::log(sum_p);
        double entropy_diff = entropy - target_entropy;

        if (std::abs(entropy_diff) < 1e-5) break;

        if (entropy_diff > 0) {
          beta_min = beta;
          beta = (beta_max == std::numeric_limits<double>::max()) ? beta * 2.0 : (beta + beta_max) / 2.0;
        } else {
          beta_max = beta;
          beta = (beta + beta_min) / 2.0;
        }
      }
    }

    m_beta(i) = beta;
  }

  for (int i = 0; i < n_samples; ++i) {
    double sum_p = 0.0;
    for (int j = 0; j < n_samples; ++j) {
      if (i != j) {
        m_probabilities_p(i, j) = std::exp(-m_beta(i) * m_distances(i, j));
        sum_p += m_probabilities_p(i, j);
      } else {
        m_probabilities_p(i, j) = 0.0;
      }
    }

    if (sum_p > 0) {
      for (int j = 0; j < n_samples; ++j) {
        m_probabilities_p(i, j) /= sum_p;
      }
    }
  }

  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_samples; ++j) {
      m_probabilities_p(i, j) = (m_probabilities_p(i, j) + m_probabilities_p(j, i)) / (2.0 * n_samples);
    }
  }
}

void TSNEAvnd::computeQDistribution()
{
  const int n_samples = m_cached_n_samples;
  double sum_q = 0.0;
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_samples; ++j) {
      if (i != j) {
        double dist = (m_embedding.row(i) - m_embedding.row(j)).squaredNorm();
        m_probabilities_q(i, j) = 1.0 / (1.0 + dist);
        sum_q += m_probabilities_q(i, j);
      } else {
        m_probabilities_q(i, j) = 0.0;
      }
    }
  }

  // Normalize
  if (sum_q > 0) {
    m_probabilities_q /= sum_q;
  }
}

void TSNEAvnd::computeGradient()
{
  const int n_samples = m_cached_n_samples;
  const int output_dims = m_cached_output_dims;

  m_gradient.setZero();

  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_samples; ++j) {
      if (i != j) {
        double p_ij = m_probabilities_p(i, j);
        double q_ij = m_probabilities_q(i, j);
        double mult = (p_ij - q_ij) * m_probabilities_q(i, j);

        for (int d = 0; d < output_dims; ++d) {
          m_gradient(i, d) += mult * (m_embedding(i, d) - m_embedding(j, d));
        }
      }
    }
    m_gradient.row(i) *= 4.0;
  }
}

double TSNEAvnd::computeKLDivergence() const
{
  const int n_samples = m_cached_n_samples;
  double kl = 0.0;

  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < n_samples; ++j) {
      if (i != j && m_probabilities_p(i, j) > 1e-12 && m_probabilities_q(i, j) > 1e-12) {
        kl += m_probabilities_p(i, j) * std::log(m_probabilities_p(i, j) / m_probabilities_q(i, j));
      }
    }
  }

  return kl;
}

}
