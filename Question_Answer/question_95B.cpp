#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cblas.h>  // BLASライブラリを使用

using namespace std;

class Mat {
    vector<double> vec;
    int cols;
    int rows;
public:
    Mat(int in_cols,int in_rows) :
        cols(in_cols), rows(in_rows),vec(in_rows*in_cols) {
        }
};

class NN {
private:
    int ind, w, w2, outd;
    double lr;
    vector<double> w2_mat, b2_mat, w3_mat, b3_mat, wout_mat, bout_mat;
    vector<double> z1, z2, z3, out;
	void dump_all(vector<double> &a) {
		cout << "#dump_all#" << endl;
		for (double& x : a) cout << x << endl;
	};
public:
    NN(int ind = 2, int w = 64, int w2 = 64, int outd = 1, double lr = 0.1)
        : ind(ind), w(w), w2(w2), outd(outd), lr(lr),
          w2_mat(ind * w), b2_mat(w),
          w3_mat(w * w2), b3_mat(w2),
          wout_mat(w2 * outd), bout_mat(outd) {

        mt19937 gen(0);
        normal_distribution<double> dist(0.0, 1.0);

        auto init = [&](vector<double>& v) {
            for (double& x : v) x = dist(gen);
        };

        init(w2_mat);
        init(b2_mat);
        init(w3_mat);
        init(b3_mat);
        init(wout_mat);
        init(bout_mat);
    }
    /*
        行列積
        A( M X K ) x ( K x N) B = C (M x N)
    */
    vector<double> matmul(const vector<double>& A, const vector<double>& B,
                        int M, int K, int N,
                        CBLAS_TRANSPOSE TranseA = CblasNoTrans,
                        CBLAS_TRANSPOSE TranseB = CblasNoTrans) {
        int lda = (TranseA == CblasNoTrans) ? K : M;
        int ldb = (TranseB == CblasNoTrans) ? N : K; 

        vector<double> C(M * N, 0.0);
                            
        cblas_dgemm(CblasRowMajor, TranseA, TranseB,
                    M, N, K, 1.0, A.data(), lda, B.data(), ldb, 0.0, C.data(), N);
        return C;
    }
    vector<double> add_bias(const vector<double>& mat, const vector<double>& bias, int rows, int cols) {
        vector<double> res = mat;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res[i * cols + j] += bias[j];
            }
        }
        return res;
    }

    vector<double> sigmoid(const vector<double>& x) {
        vector<double> res(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            res[i] = 1.0 / (1.0 + exp(-x[i]));
        }
        return res;
    }

    vector<double> forward(const vector<double>& x) {
        int data_n= x.size()/ind;
        
        z1 = x;
        z2 = sigmoid(add_bias(matmul(z1, w2_mat, data_n, ind, w), b2_mat, data_n, w));
        z3 = sigmoid(add_bias(matmul(z2, w3_mat, data_n, w, w2), b3_mat, data_n, w2));
        out = sigmoid(add_bias(matmul(z3, wout_mat, data_n, w2, outd), bout_mat, data_n, outd));
        
        return out;
    }

    void train(const vector<double>& t) {
        int data_n= t.size()/outd;
        
        ///
        vector<double> one(data_n, 1.0);
        vector<double> out_d(out.size());

        for (size_t i = 0; i < out_d.size(); i++) {
            double sig_d = out[i] * (1 - out[i]);
            out_d[i] = 2 * (out[i] - t[i]) * sig_d;
        }
        vector<double> out_dw = matmul(z3 , out_d, w2, data_n, outd  ,CblasTrans  ,CblasNoTrans);
        vector<double> out_db = matmul(one, out_d, 1 , data_n, outd,CblasNoTrans,CblasNoTrans);
        
        cblas_daxpy(w2 * outd, -lr, out_dw.data(), 1, wout_mat.data(), 1);
        cblas_daxpy(outd, -lr, out_db.data(), 1, bout_mat.data(), 1);

        ///
        vector<double> w3_d(data_n*w2);
        w3_d = matmul(out_d , wout_mat, data_n, outd, w2  ,CblasNoTrans  ,CblasTrans);
        cout << w3_d.size() << endl;
        for (size_t i = 0; i < w3_d.size(); i++) {
            w3_d[i] = w3_d[i] * (z3[i] * (1 - z3[i]));
        }
        vector<double> w3_dw = matmul(z3 , w3_d, w2, w2, data_n  ,CblasTrans  ,CblasNoTrans);
        dump_all(w3_dw);
        
    }
};

int main() {
	auto start = std::chrono::high_resolution_clock::now();
    vector<double> train_x = {0, 0, 0, 1, 1, 0, 1, 1};
    vector<double> train_t = {0, 1, 1, 0};

    NN nn(2, 64, 64, 1, 0.1);

    for (int i = 0; i < 1; i++) {
        nn.forward(train_x);
        nn.train(train_t);
    }

    // for (int i = 0; i < 4; i++) {
    //     vector<double> x = {train_x[2 * i], train_x[2 * i + 1]};
    //     vector<double> pred = nn.forward(x);
    //     cout << "in: (" << x[0] << ", " << x[1] << ") pred: " << pred[0] << endl;
    // }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "time:" << duration.count()/1000 <<  std::endl;
    return 0;
}