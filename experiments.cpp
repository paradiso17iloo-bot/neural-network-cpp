#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double rnd() { return (double)rand() / RAND_MAX * 2.0 - 1.0; }

void run(int nHid, double lr, int epochs) {
    srand(42);
    double in[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double t[4] = {0, 1, 1, 0};

    vector<vector<double>> wH(2, vector<double>(nHid));
    vector<double> bH(nHid), wO(nHid), h(nHid), dH(nHid);
    double bO = rnd();
    for (int j = 0; j < nHid; j++) {
        bH[j] = rnd(); wO[j] = rnd();
        for (int k = 0; k < 2; k++) wH[k][j] = rnd();
    }

    auto start = chrono::steady_clock::now();
    double totalError = 0;

    for (int e = 0; e <= epochs; e++) {
        totalError = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < nHid; j++) {
                double s = bH[j];
                for (int k = 0; k < 2; k++) s += in[i][k] * wH[k][j];
                h[j] = sigmoid(s);
            }
            double so = bO;
            for (int j = 0; j < nHid; j++) so += h[j] * wO[j];
            double out = sigmoid(so);

            double err = t[i] - out;
            totalError += err * err;
            double dO = err * out * (1 - out);
            for (int j = 0; j < nHid; j++) dH[j] = dO * wO[j] * h[j] * (1 - h[j]);
            for (int j = 0; j < nHid; j++) {
                wO[j] += lr * dO * h[j];
                bH[j] += lr * dH[j];
                for (int k = 0; k < 2; k++) wH[k][j] += lr * dH[j] * in[i][k];
            }
            bO += lr * dO;
        }
    }

    double ms = chrono::duration<double, milli>(chrono::steady_clock::now() - start).count();

    int correct = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < nHid; j++) {
            double s = bH[j];
            for (int k = 0; k < 2; k++) s += in[i][k] * wH[k][j];
            h[j] = sigmoid(s);
        }
        double so = bO;
        for (int j = 0; j < nHid; j++) so += h[j] * wO[j];
        if ((sigmoid(so) > 0.5) == (t[i] > 0.5)) correct++;
    }

    cout << setw(8) << nHid << setw(8) << lr << setw(10) << epochs
         << setw(14) << totalError << setw(10) << correct << "/4"
         << setw(12) << ms << " ms" << endl;
}

int main() {
    cout << setw(8) << "Hidden" << setw(8) << "LR" << setw(10) << "Epochs"
         << setw(14) << "FinalError" << setw(12) << "Correct" << setw(15) << "Time" << endl;

    run(4, 0.5,  20000);   
    run(2, 0.5,  20000);
    run(1, 0.5,  20000);
    run(8, 0.5,  20000);
    run(4, 0.05, 20000);
    run(4, 0.5,  2000);    
}