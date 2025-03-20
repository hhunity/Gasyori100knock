#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cblas.h>  // BLASライブラリを使用
#include <iomanip>
#include <memory>

using namespace std;

void debug(string str="") {
    cout << "debug:" << str << endl;
}

template <typename t>
class Mat  {
    unique_ptr<t[]> vec;
    int rows;
    int cols;
    string name;
    void _set_size(int in_rows,int in_cols){rows=in_rows,cols=in_cols;}
    unique_ptr<t[]>  _set_mem(int size){return make_unique<t[]>(size);}
public:
    Mat(string name="") : rows(0),cols(0),name(name){}
    Mat(int in_rows,int in_cols,string name="") : rows(in_rows),cols(in_cols),vec(_set_mem(in_rows*in_cols)),name(name) {}
    Mat(Mat&& other) noexcept : rows(other.rows),cols(other.cols),vec(std::move(other.vec)),name(other.name) {}

    auto begin() { return vec.get();}
    auto end() {return vec.get()+size();}
    auto begin() const { return vec.get();}
    auto end() const {return vec.get()+size();}
    
    t* data() {return vec.get();}
    const t* data() const {return vec.get();}

    t& operator()(int in_rows,int in_cols){return vec[cols*in_rows+in_cols];}
    const t& operator()(int in_rows,int in_cols) const {return vec[cols*in_rows+in_cols];}

    t& operator[](int index) {return vec[index];}
    const t& operator[](int index) const {return vec[index];}
    
    void set_data(t data) {std::fill(vec.get(),vec.get()+size(),data);}
    bool set_size(int in_rows,int in_cols){
        if(in_rows!=rows or in_cols!=cols) {
            _set_size(in_rows,in_cols);
            vec = _set_mem(in_rows*in_cols);
            return true;
        }
        return false;
    }
    void copy_from_vector(const vector<vector<t>>& other) {
        set_size(other.size(),other[0].size());
        for (int j=0;j<rows;j++) {
            for( int i=0;i<cols;i++) {
              vec[j*cols+i] = other[j][i];
            }
        }
    }
    int  get_rows() const {return rows;}
    int  get_cols() const {return cols;}
    int  size() const {return rows*cols;}
    void dump_all() const {
        dump_col_row("dump_all start");
        for (int c =0;c<size();c++) {
            cout << internal << setw(8) << fixed << setprecision(5) << vec[c] << " ";
            if ( (c+1)%cols == 0) cout << endl;
        }
        dump_col_row("dump_all end");
    }
	void dump_col_row(const string str="") const {
		cout << name << ":"<< rows << "x" << cols <<":"<< str << endl; 
	}
};

class NN {
private:
    double lr;
    Mat<double> w2_mat, b2_mat, w3_mat, b3_mat, wout_mat, bout_mat;
    Mat<double> z1, z2, z3, out;
    Mat<double> one,out_d,out_dw,out_db,one2,w3_d,w3_d2,w3_dw,w3_db,w2_d,w2_dw,w2_db;
    /*
        行列積
        A( M X K ) x ( K x N) B = C (M x N)
        http://azalea.s35.xrea.com/blas/gemm.html
    */
    Mat<double>& matmul(const Mat<double>& A, const Mat<double>& B,Mat<double>& C,
                        CBLAS_TRANSPOSE TranseA = CblasNoTrans,CBLAS_TRANSPOSE TranseB = CblasNoTrans) {
        //転置後の行列を求める
        int Arows = (TranseA == CblasNoTrans) ? A.get_rows() : A.get_cols();
        int Acols = (TranseA == CblasNoTrans) ? A.get_cols() : A.get_rows();
        int Brows = (TranseB == CblasNoTrans) ? B.get_rows() : B.get_cols();
        int Bcols = (TranseB == CblasNoTrans) ? B.get_cols() : B.get_rows();
        
        C.set_size(Arows,Bcols);
        
        cblas_dgemm(CblasRowMajor, TranseA, TranseB,
            Arows, Bcols, Acols, 1.0, A.data(), A.get_cols(), B.data(), B.get_cols(), 0.0, C.data(), Bcols);
        
        return C;
    }
    Mat<double> add_bias(const Mat<double>& mat, const Mat<double>& bias) {
        Mat<double> res(mat.get_rows(),mat.get_cols());
        
        for (int i = 0; i < mat.get_rows(); i++) {
            for (int j = 0; j < mat.get_cols(); j++) {
                res(i,j) = mat(i,j) + bias[j];
            }
        }
        
        return res;
    }
    
    Mat<double>& sum(const Mat<double>& mat1,Mat<double>& out,double alpha) {
        cblas_daxpy(out.size(),alpha, mat1.data(), 1, out.data(), 1);
        return out;
    }

    void sigmoid(const Mat<double>& x,Mat<double> &out) {
        out.set_size(x.get_rows(),x.get_cols());
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = 1.0 / (1.0 + exp(-x[i]));
        }
    }
    
public:
    NN(int ind = 2, int w = 64, int w2 = 64, int outd = 1, double lr = 0.1)
        : lr(lr),
          w2_mat(ind,w,"w2"), b2_mat(1,w,"b2"),w3_mat(w,w2,"w3"), b3_mat(1,w2,"b3"),
          wout_mat(w2,outd,"wout"), bout_mat(outd,1,"bout"),
          z1("z1"),z2("z2"),z3("z3"),one("one"),out_d("out_d"),
          out_dw("out_dw"),out_db("out_db"),one2("one2"),w3_d("w3_d"),w3_d2("w3_d2"),w3_db("w3_db"),
          w3_dw("w3_dw"),w2_d("w2_d"),w2_dw("w2_dw"),w2_db("w2_db")
          {

        mt19937 gen(0);
        normal_distribution<double> dist(0.0, 1.0);
        
        auto init = [&](Mat<double>& v) {
            for (double& x : v) x = dist(gen);
        };
        
        init(w2_mat);
        init(b2_mat);
        init(w3_mat);
        init(b3_mat);
        init(wout_mat);
        init(bout_mat);
    }
    vector<double> forward(const vector<vector<double>>& x) {
        
        z1.copy_from_vector(x);
        sigmoid(add_bias(matmul(z1,w2_mat,z2), b2_mat),z2);
        sigmoid(add_bias(matmul(z2,w3_mat,z3), b3_mat),z3);
        sigmoid(add_bias(matmul(z3,wout_mat,out), bout_mat),out);

        vector<double> out_vec(out.size());
        for(int i=0;i<out.size();i++) {out_vec[i]=out[i];}
        return out_vec;
    }

    void train(const vector<double>& t) {
        int data_n= t.size();
        
        ///
        if(one.set_size(data_n,1)){
            one.set_data(1.0);
        }
        out_d.set_size(out.get_rows(),out.get_cols());
        
        for (size_t i = 0; i < out_d.size(); i++) {
            double sig_d = out[i] * (1 - out[i]);
            out_d[i] = 2 * (out[i] - t[i]) * sig_d;
        }
        matmul(z3 , out_d, out_dw, CblasTrans  ,CblasNoTrans);
        matmul(one, out_d, out_db, CblasTrans  ,CblasNoTrans);
        sum(out_dw,wout_mat,-lr);
        sum(out_db,bout_mat,-lr);
        
        ///
		if( one2.set_size(z3.get_rows(),z3.get_cols()) ) {
            one2.set_data(1.0);
        }
        matmul(out_d , wout_mat,w3_d , CblasNoTrans  ,CblasTrans);
        for (size_t i = 0; i < w3_d.size(); i++) {
            w3_d[i] = w3_d[i] * (z3[i] * (1 - z3[i]));
        }
        matmul(z3 ,w3_d,w3_dw,CblasTrans ,CblasNoTrans);
        matmul(one2,w3_d,w3_db,CblasTrans ,CblasNoTrans);
        sum(w3_dw,w3_mat,-lr);
        sum(w3_db,b3_mat,-lr);

        ///
        matmul(w3_d , w3_mat,w2_d,CblasNoTrans  ,CblasTrans);
        for (size_t i = 0; i < w2_d.size(); i++) {
            w2_d[i] = w2_d[i] * (z2[i] * (1 - z2[i]));
        }
        matmul(z1   , w2_d,w2_dw,CblasTrans ,CblasNoTrans);
        matmul(one  , w2_d,w2_db,CblasTrans ,CblasNoTrans);
        sum(w2_dw,w2_mat,-lr);
        sum(w2_db,b2_mat,-lr);        
    }
};

int main() {
	auto start = std::chrono::high_resolution_clock::now();
    vector<vector<double>> train_x = {{0, 0}, {0, 1},{1, 0},{1, 1}};
    vector<double> train_t = {0, 1, 1, 0};

    NN nn(2, 64, 64, 1, 0.1);

    for (int i = 0; i < 1000; i++) {
        nn.forward(train_x);
        nn.train(train_t);
    }

    for (int i = 0; i < 4; i++) {
        vector<vector<double>> x={{train_x[i][0],train_x[i][1]}};
        vector<double> pred = nn.forward(x);
        cout << "in: (" << x[0][0] << ", " << x[0][1] << ") pred: " << pred[0] << endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "time:" << duration.count()/1000 <<  std::endl;

    return 0;
}