#pragma once

#include <halp/controls.hpp>
#include <halp/meta.hpp>
#include <vector>
#include <Eigen/Dense>

namespace puara_gestures::objects
{

class TSNEAvnd
{
public:
  halp_meta(name, "t-SNE")
  halp_meta(category, "ML/Puara")
  halp_meta(c_name, "puara_tsne_avnd")
  halp_meta(description, "Performs t-SNE dimensionality reduction on a dataset.")
  halp_meta(uuid, "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d")

  struct ins
  {
    halp::val_port<"Data", std::vector<double>> data;
    halp::knob_i32<"Num Features", halp::range{1, 128, 2}> n_features;
    halp::knob_i32<"Output Dims", halp::range{1, 3, 2}> output_dims;
    halp::knob_f32<"Perplexity", halp::range{5.0, 50.0, 30.0}> perplexity;
    halp::knob_f32<"Learning Rate", halp::range{10.0, 1000.0, 200.0}> learning_rate;
    halp::knob_i32<"Max Iterations", halp::range{100, 2000, 1000}> max_iterations;
    halp::impulse_button<"Reset"> reset;
  } inputs;

  struct outs
  {
    halp::val_port<"Embedded Data", std::vector<double>> embedded_data;
    halp::val_port<"KL Divergence", double> kl_divergence;
  } outputs;

  void operator()();

private:
  std::vector<double> m_embedded_data;
  bool m_is_computed{false};
  int m_current_iteration{0};

  // Pre-allocated matrices for memory reuse
  mutable Eigen::MatrixXd m_data_matrix;
  mutable Eigen::MatrixXd m_distances;
  mutable Eigen::MatrixXd m_probabilities_p;
  mutable Eigen::MatrixXd m_probabilities_q;
  mutable Eigen::MatrixXd m_embedding;
  mutable Eigen::MatrixXd m_gradient;
  mutable Eigen::MatrixXd m_velocity;
  mutable Eigen::VectorXd m_beta;
  mutable Eigen::VectorXd m_entropy;

  // Cache dimensions
  mutable int m_cached_n_samples{0};
  mutable int m_cached_n_features{0};
  mutable int m_cached_output_dims{0};

  // Helper functions
  void computeGaussianKernel();
  void computeQDistribution();
  void computeGradient();
  double computeKLDivergence() const;
};

}
