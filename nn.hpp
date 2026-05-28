#ifndef NN_HPP
#define NN_HPP

#include <vector>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <sstream>
#include "layer.hpp"
#include "matrix.hpp"
#include "log.hpp"

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

      void train(const std::vector<std::vector<double>>& inputs,
                 const std::vector<std::vector<double>>& targets,
                 double learningRate,
                 size_t epochs,
                 size_t batchSize = 1,
                 bool verbose = true) {
                
          if(inputs.size() != targets.size()) {
              throw std::runtime_error("Input and target sizes don't match");
          }
        
          if(learningRate <= 0 || epochs <= 0 || batchSize <= 0) {
              throw std::runtime_error("Learning rate, epochs and batch size must be positive");
          }
        
          size_t datasetSize = inputs.size();
          auto start = std::chrono::high_resolution_clock::now();
          double totalError = 0.;
          double baseLR = learningRate;
          double decayLR = 0.996; //0.4% por epoca
          double minLR = 1e-4;
        
          if(verbose) {
              std::stringstream data;
              data << "--------------------- dados do treinamento -------------------\n";
              data << "taxa de aprendizado = " << learningRate
                   << "\t taxa de decaimento = "<<(1.0 - decayLR)*100<<"%"
                   << "\t EPOCAS = " << epochs <<"\n";
          
              data << "tamanho dos dados = " << datasetSize 
                   << "\t tamanho do batch = " << batchSize << "\n";
              data << "\n------------------------ treinando ... -----------------------\n";            
              std::cout << data.str();
              L::log(data.str());
          }
        
          for(int e=0; e<epochs; e++) {
              learningRate = std::max(minLR, baseLR * std::pow(decayLR, e)); // diminui com as epocas
              totalError = 0.;
          
              for(size_t b=0; b<datasetSize; b+=batchSize) {
                  size_t actualBatchSize = std::min(batchSize, (datasetSize - b));
              
                  std::vector<Matrix> weightBGrads;
                  for(size_t i=0; i<this->weights.size(); i++) {
                      weightBGrads.emplace_back(this->weights[i].getRows(), this->weights[i].getCols());
                  }
                
                  std::vector<std::vector<double>> biasBGrads(this->layers.size() - 1);
                  for(size_t i=0; i<biasBGrads.size(); i++) {
                      biasBGrads[i].resize(this->layers[i+1].size, 0.);
                  }
                
                  for(size_t k = b; k < (b + actualBatchSize); k++) {
                      foward(inputs[k]);
                      Layer& outputLayer = this->layers.back();
                  
                      const double eps = 1e-7;
                      for(size_t i=0; i<outputLayer.size; i++) {
                          double yTrue = targets[k][i];
                          double yPred = outputLayer.after[i];
                      
                          // como n tem log(0) precisamos disso
                          yPred = std::min(std::max(yPred, eps), 1.0 - eps);
                      
                          // binary cross-entropy loss (BCE)
                          totalError += - (yTrue * std::log(yPred) + (1.0 - yTrue) * std::log(1.0 - yPred));
                      
                          outputLayer.gradient[i] = yPred - yTrue;
                      }
                    
                      for (int l = (static_cast<int>(this->layers.size()) - 2); l > 0; l--) {
                          for (size_t i = 0; i < this->weights[l].getRows(); ++i) {
                              double error = 0.;
                              for (size_t j = 0; j < this->weights[l].getCols(); ++j) {
                                  error += this->layers[l+1].gradient[j] * this->weights[l](i, j);
                              }
                              layers[l].gradient[i] = error * layers[l].applyDActivation(layers[l].before[i]);
                          }
                      }
                    
                      for (int l = (static_cast<int>(layers.size()) - 2); l >= 0; l--) {
                          const std::vector<double>& activations = (l == 0) ? inputs[k] : layers[l].after;
                      
                          for (size_t i = 0; i < weightBGrads[l].getRows(); ++i) {
                              for (size_t j = 0; j < weightBGrads[l].getCols(); ++j) {
                                  weightBGrads[l](i, j) += layers[l+1].gradient[j] * activations[i];
                              }
                          }
                        
                          for (size_t i = 0; i < layers[l+1].size; ++i) {
                              biasBGrads[l][i] += layers[l+1].gradient[i];
                          }
                      }
                  }
                
                  for (int l = (static_cast<int>(layers.size()) - 2); l >= 0; l--) {
                      for (size_t i = 0; i < weights[l].getRows(); ++i) {
                          for (size_t j = 0; j < weights[l].getCols(); ++j) {
                              weights[l](i, j) -= learningRate * (weightBGrads[l](i,j) / actualBatchSize);
                          }
                      }
                    
                      for (size_t i = 0; i < layers[l+1].size; ++i) {
                          layers[l+1].bias[i] -= learningRate * (biasBGrads[l][i] / actualBatchSize);
                      }
                  }
              }
            
              if(verbose && e % 100 == 0) {
                  std::stringstream data;
                  data << "["
                       << (100 * e / epochs) << "%] EPOCA : " << e
                       << " | BCE: " << totalError / inputs.size() << "\n";
                  std::cout << data.str();
                  L::log(data.str());
              }
          }
        
          auto end = std::chrono::high_resolution_clock::now();
          if(verbose) {
              std::stringstream data;
              data << "------------------- treino completo ---------------------\n";
              data << "BCE final : " << totalError / inputs.size() << "\n";
              data << "Tempo de treino : "
                   << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                   << " ms\n";
              std::cout << data.str();
              L::log(data.str());
          }
      }

       void saveModel(const std::string& filename = "model.csv") const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Unable to open file: " << filename << std::endl;
                return;
            }

            file << "type,layer,row,col,value\n"; 

            for (size_t l = 1; l < layers.size();++l)
            {
                for (size_t i = 0; i < weights[l-1].getRows();++i)
                {
                    for (size_t j = 0; j < weights[l-1].getCols();++j)
                    {
                        file << "weight," << l << "," << i << "," << j << "," <<  weights[l-1](i,j) << "\n";
                    }
                }

                for (size_t i = 0; i < layers[l].bias.size();++i)
                {
                    file << "bias," << l << "," << i << ",0," << layers[l].bias[i] << "\n";
                }
            }

            file.close();
            std::cout << "Model saved to " << filename << std::endl;
        }

        void loadModel(const std::string& filename = "model.csv") {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Unable to open model file: " << filename << std::endl;
                return;
            }

            std::string line;
            std::getline(file, line);

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string type, layerStr, rowStr, colStr, valueStr;

                std::getline(ss, type, ',');
                std::getline(ss, layerStr, ',');
                std::getline(ss, rowStr, ',');
                std::getline(ss, colStr, ',');
                std::getline(ss, valueStr, ',');

                int layer = std::stoi(layerStr);
                int row = std::stoi(rowStr);
                int col = std::stoi(colStr);
                double value = std::stod(valueStr);

                if (type == "weight") {
                    weights[layer-1](row, col) = value;
                } else if (type == "bias") {
                    layers[layer].bias[row] = value;
                }
            }

            file.close();
            std::cout << "Model loaded from " << filename << std::endl;
        }
};

#endif