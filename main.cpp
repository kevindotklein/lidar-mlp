#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include "layer.hpp"
#include "nn.hpp"
#include "csv.hpp"
#include "log.hpp"
#include <sstream>

using namespace std;

int main(){

    try{
        vector<Layer> layers;

        
        layers.emplace_back(0,24,ActivationType::NONE);
        layers.emplace_back(1, 12, ActivationType::RELU);
        layers.emplace_back(2, 1, ActivationType::SIGMOID);

        NeuralNetwork nn(layers);

        vector<vector<double>> trainingFeatures;
        vector<vector<double>> trainingLabels;
        vector<int> training_ids;

        vector<vector<double>> test_features;
        vector<vector<double>> test_labels;
        vector<int> test_ids;

        string load_model;

        if (CSV::loadAndSplitSensorData("sensor_readings_24.csv",
                                    trainingFeatures, trainingLabels, training_ids,
                                    test_features, test_labels, test_ids,0.8)) {
            
            cout << "Deseja carregar o modelo? (s/n): ";
            cin >> load_model;
            if (load_model == "s" || load_model == "S") {
                cout << "carregando modelo...\n";
                nn.loadModel();
            } else {
                load_model = "n";
                nn.train(trainingFeatures, trainingLabels, 0.029, 1300,8);
            }                          
            

        }else{
            cerr << "Failed to load sensor data\n";
            return 1; 
        }
        
        
        cout << "-------------------Predicoes--------------------\n";
        L::log("-------------------Predicoes--------------------\n");
        for (size_t i = 0; i < test_features.size(); ++i) {
            stringstream data;
            data << "-------------------Predicao["<<i<<"]--------------------\n"; 
            data << "Row ID     : " << test_ids[i] << "\n";

            data << "Dados: ";
            for (double val : test_features[i]) {
                data << fixed << setprecision(2) << val << " ";
            }
            data << "\n";

            vector<double> prediction = nn.predict(test_features[i]);

            data << "Predicao: " << fixed << setprecision(4) << prediction[0] << "\n";
            data << "Verdadeiro:     : " << test_labels[i][0] << "\n";

            cout << data.str();
            L::log(data.str());
        }

        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (size_t i = 0; i < test_features.size(); ++i) {
            double pred = nn.predict(test_features[i])[0];
            int predicted = pred >= 0.5 ? 1 : 0;
            int actual = static_cast<int>(test_labels[i][0]);

            if (predicted == 1 && actual == 1) tp++;
            else if (predicted == 0 && actual == 0) tn++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 1) fn++;
        }

        int total = tp + tn + fp + fn;
        double accuracy = static_cast<double>(tp + tn) / total;

        stringstream data;
        data << "-------------------Metricas---------------------\n";
        data << "\nConfusion Matrix:\n";
        data << "TP: " << tp << " | FP: " << fp << "\n";
        data << "FN: " << fn << " | TN: " << tn << "\n";

        data << std::fixed << std::setprecision(4);
        data << "\nAcuracia : " << accuracy * 100 << "%\n";
        cout << data.str();
        L::log(data.str());

        if(load_model == "n"){
            string save_model;
            cout << "Deseja salvar o modelo? (s/n): ";
            cin >> save_model;
            if (save_model == "s" || save_model == "S") {
                cout << "Salvando o modelo...\n";
                nn.saveModel();
            } else {
                cout << "Model not saved.\n";
            }
        }
        
    }catch(const exception &e){
        cerr << " ERROR : " << e.what() << endl;
        return 1;
    }

    return 0;
}
