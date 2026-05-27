#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>

using ActivationFunction = std::function<double(double)>;

namespace Activation {
  inline double relu(double x) { return (x > 0.f) ? x : 0.f; }
  inline double dRelu(double x) { return (x > 0.f) ? 1.f : 0.f; }
  inline double sigmoid(double x) { return 1.f / (1.f + std::exp(-x)); }
  inline double dSigmoid(double x) {
    double s = sigmoid(x);
    return s * (1.f - s);
  }
}

enum class ActivationType {
  NONE,
  RELU,
  SIGMOID
};

inline std::pair<ActivationFunction, ActivationFunction>
getActivationPair(ActivationType type) {
  using namespace Activation;
  switch (type)
  {
  case ActivationType::RELU:
    return {relu, dRelu};
  case ActivationType::SIGMOID:
    return {sigmoid, dSigmoid};
  case ActivationType::NONE:
  default:
    return {ActivationFunction{}, ActivationFunction{}};
  }
}

class Layer {
  private:
    ActivationFunction activation;
    ActivationFunction dActivation;

  public:
    int layerIndex; // posicao da camada na rede
    int size; // quantos neuronios na camada
    std::vector<double> before; // antes de aplicar na funcao
    std::vector<double> after; // depois de aplicar na funcao
    std::vector<double> bias;
    std::vector<double> gradient;

    Layer() {}
    Layer(int index, int size, ActivationType activationType)
    : layerIndex(index), size(size), before(size, 0.), after(size, 0.) {

      if(size <= 0) {
        throw std::invalid_argument("Layer size must be positive");
      }

      if(index != 0) {
        this->gradient = std::vector<double>(size, 0.);
        this->bias = std::vector<double>(size, 0.);


        this->activation = getActivationPair(activationType).first;
        this->dActivation = getActivationPair(activationType).second;
      }
    }

    ActivationFunction getActivation() const {
      return this->activation;
    }

    ActivationFunction getDActivation() const {
      return this->dActivation;
    }

    bool hasActivation() const {
      return static_cast<bool>(this->activation);
    }

    bool hasDActivation() const {
      return static_cast<bool>(this->dActivation);
    }

    double applyActivation(double x) const {
      if(!this->activation) {
        throw std::runtime_error("This layer has no activation function");
      }

      return this->activation(x);
    }

    double applyDActivation(double x) const {
      if(!this->dActivation) {
        throw std::runtime_error("This layer has no dActivation function");
      }

      return this->dActivation(x);
    }
};

#endif