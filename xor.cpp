#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace std;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

const int N_IN  = 2;   
const int N_HID = 4;   

double randomWeight() { return (double)rand() / RAND_MAX * 2.0 - 1.0; } 

int main() {
    srand(42);

    double inputs[4][N_IN] = {{0,0},{0,1},{1,0},{1,1}};
    double targets[4]      = {0, 1, 1, 0};   

    double wH[N_IN][N_HID];   
    double bH[N_HID];         
    double wO[N_HID];         
    double bO = randomWeight();

    for (int j = 0; j < N_HID; j++) {
        bH[j] = randomWeight();
        wO[j] = randomWeight();
        for (int k = 0; k < N_IN; k++) wH[k][j] = randomWeight();
    }

    double lr = 0.05;
    int epochs = 20000;

    for (int epoch = 0; epoch <= epochs; epoch++) {
        double totalError = 0;

        for (int i = 0; i < 4; i++) {
            double h[N_HID];
            for (int j = 0; j < N_HID; j++) {
                double sum = bH[j];
                for (int k = 0; k < N_IN; k++) sum += inputs[i][k] * wH[k][j];
                h[j] = sigmoid(sum);
            }
            double sumO = bO;
            for (int j = 0; j < N_HID; j++) sumO += h[j] * wO[j];
            double out = sigmoid(sumO);

            double error = targets[i] - out;
            totalError += error * error;

            double dOut = error * out * (1 - out);
            double dHid[N_HID];
            for (int j = 0; j < N_HID; j++)
                dHid[j] = dOut * wO[j] * h[j] * (1 - h[j]);  

            for (int j = 0; j < N_HID; j++) {
                wO[j] += lr * dOut * h[j];
                bH[j] += lr * dHid[j];
                for (int k = 0; k < N_IN; k++)
                    wH[k][j] += lr * dHid[j] * inputs[i][k];
            }
            bO += lr * dOut;
        }

        if (epoch % 2000 == 0)
            cout << "Epoch " << epoch << "  error = " << totalError << endl;
    }

    cout << "\nResults:\n";
    for (int i = 0; i < 4; i++) {
        double h[N_HID];
        for (int j = 0; j < N_HID; j++) {
            double sum = bH[j];
            for (int k = 0; k < N_IN; k++) sum += inputs[i][k] * wH[k][j];
            h[j] = sigmoid(sum);
        }
        double sumO = bO;
        for (int j = 0; j < N_HID; j++) sumO += h[j] * wO[j];
        cout << inputs[i][0] << " XOR " << inputs[i][1]
             << " = " << sigmoid(sumO) << endl;
    }
}