#include <iostream>
#include <cmath>
using namespace std;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

int main() {
    double inputs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double targets[4]   = {0, 0, 0, 1};   
    double w1 = 0.5, w2 = -0.4, bias = 0.1;
    double lr = 0.5;                       

    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < 4; i++) {
            double out = sigmoid(inputs[i][0]*w1 + inputs[i][1]*w2 + bias);
            double error = targets[i] - out;
            double grad = error * out * (1 - out);
            w1   += lr * grad * inputs[i][0];
            w2   += lr * grad * inputs[i][1];
            bias += lr * grad;
        }
    }
    for (int i = 0; i < 4; i++) {
        double out = sigmoid(inputs[i][0]*w1 + inputs[i][1]*w2 + bias);
        cout << inputs[i][0] << " AND " << inputs[i][1] << " = " << out << endl;
    }
}