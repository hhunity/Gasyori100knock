#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <string>

using namespace std;

using Mat = Eigen::MatrixXd;

void dump(Mat mat) {
		cout << mat << endl; 
		cout << mat.rows() << "x" << mat.cols() ;
}

class NN {
	Mat w2,b2,w3,b3,wout,bout;
	Mat z1,z2,z3,out;
	double lr;
private:
	void set_all(Mat &a,function<double()> func) {
		for(int y=0;y<a.rows();y++) {
			for(int x=0;x<a.cols();x++) {
				a(y,x) = func();
			}
		}
	}
	void dump_all(const Mat&a) {
		cout << "#dump_all#" << endl;
		cout << a << endl;
	}
	void dump_col_row(const Mat &a,string str) {
		cout << str << ":"<< a.rows() << "x" << a.cols() << endl; 
	}
	void dump_col_row(const Mat &a) {
		cout << a.rows() << "x" << a.cols() << endl; 
	}
	Mat sigmoid(const Mat& x) {
		return 1.0 / (1.0 + (-x.array()).exp());
	}
public:
	NN(int ind=2,int iw=64,int iw2=64,int outd=1,double lr=0.1) :
		w2{ind,iw},b2{1,iw},w3{iw,iw2},b3{1,iw2},wout{iw2,outd},bout{outd,1},lr(lr)
	{
		mt19937 gen(0);
		normal_distribution<> dist(0.0,1.0);
		
		set_all(w2,[&dist,&gen](){return dist(gen);});
		set_all(b2,[&dist,&gen](){return dist(gen);});
		set_all(w3,[&dist,&gen](){return dist(gen);});
		set_all(b3,[&dist,&gen](){return dist(gen);});
		set_all(wout,[&dist,&gen](){return dist(gen);});
		set_all(bout,[&dist,&gen](){return dist(gen);});
	}
	Mat forward(const Mat &x) {

		int num = x.rows();
		z1 = x;
		z2 = sigmoid(((z1*w2)+b2.replicate(num, 1)));
		z3 = sigmoid(((z2*w3)+b3.replicate(num, 1)));
		out= sigmoid(((z3*wout)+bout.replicate(num,1)));

		return out;
	}
	void train(Mat &t) {
		int num = t.rows();
		Mat one(num,1);
		one.setOnes();//1で埋める

		//self.out * (1 - self.out)はシグモンドレイヤの逆伝搬
		//2*(self.out - t) は２乗誤差の逆伝搬
		Mat out_d = 2*(out.array() - t.array()) * out.array() * (one.array() - out.array());
		//ffineの逆伝搬
		Mat out_dw= z3.transpose()*out_d;
		//バイアスの逆伝搬
		Mat out_db= one.transpose()*out_d;
		wout.array() -= (lr * out_dw.array());
		bout.array() -= (lr * out_db.array());

		Mat one2(z3.rows(),z3.cols());
		one2.setOnes();
		Mat w3_d = (out_d * wout.transpose()).array() * z3.array() * (one2.array()- z3.array());
		Mat w3_dw= z3.transpose()*w3_d;
		Mat w3_db= one.transpose()*w3_d;
		w3.array() -= (lr * w3_dw.array());
		b3.array() -= (lr * w3_db.array());

		Mat w2_d = (w3_d * w3.transpose()).array() * z2.array() * (one2.array()- z2.array());
		Mat w2_dw= z1.transpose()*w2_d;
		Mat w2_db= one.transpose()*w2_d;
		w2.array() -= (lr * w2_dw.array());
		b2.array() -= (lr * w2_db.array());

		if (out_d.hasNaN() || out_dw.hasNaN() || out_db.hasNaN()) {
			cerr << "NaN detected in gradient calculation!" << endl;
		}
	}
};

double get_correct_coutn(Mat &out,Mat &train_y){
	int correct_count = 0;
	for(int i = 0 ;i< 4;i++ ){
		if(train_y(i) == 1 && ( out(i,0) >= 0.5) ) {
			correct_count++;
		}
		if(train_y(i) == 0 && ( out(i,0) < 0.5) ) {
			correct_count++;
		}
	}

	return correct_count;
}

int main() {
	auto start = chrono::high_resolution_clock::now();

	Mat train_x(4,2);
	Mat train_y(4,1);
	train_x << 0,0,
						 0,1,
						 1,0,
						 1,1;
	train_y << 0,1,1,0;

	NN nn{2,64,64,1,0.1};

	for( int i =0;i <1000;i++) {
		Mat out = nn.forward(train_x);
		nn.train(train_y);
		// cout << (double)get_correct_coutn(out,train_y)/4 << endl;
	};

	for(int i=0;i<4;i++)	{
		Mat x = train_x.row(i);
		Mat out = nn.forward(x);
		std :: cout << "in:" << x << "pred:" << out << endl;
	}

  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> duration = end - start;
	cout << "time:" << duration.count()/1000 <<  endl;
}
