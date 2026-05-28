#ifndef NN_HPP
#define NN_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include "layer.hpp"
#include "matrix.hpp"

class NeuralNetwork {
  private:
    std::vector<Layer> layers;
    std::vector<Matrix> weights;

    void connectLayers() {
      for(size_t i=0; i<this->layers.size() - 1; i++) {
        this->weights.emplace_back(this->layers[i].size, layers[i + 1].size);
      }
    }

    void initWeights() {
      for(auto& ws : this->weights) {
        ws.fillRandom();

        // he initialization (é bom pra nns q usam relu normalmente)
        ws *= sqrt(2. / ws.getRows());
      }
    }
    
    public:
      NeuralNetwork() {}
      NeuralNetwork(const std::vector<Layer>& layers) : layers(layers) {
        this->connectLayers();
        this->initWeights();
      }

      std::vector<double> foward(const std::vector<double>& input) {
        if(input.size() != this->layers[0].size) {
          throw std::runtime_error("Input size mismatch");
        }

        this->layers[0].after = input;

        for(size_t i = 0; i < this->layers.size() - 1; i++) {
          for(size_t j=0; j<this->weights[i].getCols(); j++) {
            double s = this->layers[i+1].bias[j];

            for(size_t k=0; k<this->weights[i].getRows(); k++) {
              s += this->layers[i].after[k] * this->weights[i](k, j);
            }

            this->layers[i+1].before[j] = s;
            this->layers[i+1].after[j] = this->layers[i+1].applyActivation(s);
          }
        }

        return this->layers.back().after;
      }

      std::vector<double> predict(const std::vector<double>& input) {
        return this->foward(input);
      }
};

#endif