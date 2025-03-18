#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <string>

using Mat = Eigen::MatrixXd;

class NN {

	Mat w2,b2,w3,b3,wout,bout;
	Mat z1,z2,z3,out;
	double lr;

private:
	void set_all(Mat &a,std::function<double()> func) {
		for(int y=0;y<a.rows();y++) {
			for(int x=0;x<a.cols();x++) {
				a(y,x) = func();
			}
		}
	};
	void dump_all(const Mat&a) {
		std::cout << "#dump_all#" << std::endl;
		std::cout << a << std::endl;
		// for(int y=0;y<a.rows();y++) {
		// 	for(int x=0;x<a.cols();x++) {
		// 		std::cout << y << "," << x << ":" << a(y,x) << std::endl;
		// 	}
		// }
	};
	void dump_col_row(const Mat &a,std::string str) {
		std::cout << str << ":"<< a.rows() << "x" << a.cols() << std::endl; 
	};
	void dump_col_row(const Mat &a) {
		std::cout << a.rows() << "x" << a.cols() << std::endl; 
	};
	Mat sigmoid(const Mat& x) {
		// std::cout<< x.array() << std::endl;
		return 1.0 / (1.0 + (-x.array()).exp());
	};
public:
	NN(int ind=2,int iw=64,int iw2=64,int outd=1,double lr=0.1) :
		w2{ind,iw},b2{1,iw},w3{iw,iw2},b3{1,iw2},wout{iw2,outd},bout{outd,1},lr(lr)
	{
		std::mt19937 gen(0);
		std::normal_distribution<> dist(0.0,1.0);
		
		set_all(w2,[&dist,&gen](){return dist(gen);});
		set_all(b2,[&dist,&gen](){return dist(gen);});
		set_all(w3,[&dist,&gen](){return dist(gen);});
		set_all(b3,[&dist,&gen](){return dist(gen);});
		set_all(wout,[&dist,&gen](){return dist(gen);});
		set_all(bout,[&dist,&gen](){return dist(gen);});

		// dump_all(w2);
		// dump_all(b2);
		// dump_all(w3);
		// dump_all(b3);
		// dump_all(wout);
		// dump_all(bout);

	};
	Mat forward(const Mat &x) {

		int num = x.rows();
		z1 = x;
		z2 = sigmoid(((z1*w2)+b2.replicate(num, 1)));
		z3 = sigmoid(((z2*w3)+b3.replicate(num, 1)));
		out= sigmoid(((z3*wout)+bout.replicate(num,1)));

		return out;
	};
	void test(){

	};
	void train(Mat &t) {
		int num = t.rows();
		Mat one(num,1);
		one.setOnes();//1で埋める

		//self.out * (1 - self.out)はシグモンドレイヤの逆伝搬
		//2*(self.out - t) は２乗誤差の逆伝搬
		Mat out_d = 2*(out.array() - t.array()) * out.array() * (one - out).array();
		Mat out_dw= z3.transpose()*out_d;
		Mat out_db= one.transpose()*out_d;
		// wout.array() -= (lr * out_dw.array());
		// bout.array() -= (lr * out_db.array());
		wout -= (lr * out_dw);
		bout -= (lr * out_db);

		Mat one2(z3.rows(),z3.cols());
		Mat w3_d = (out_d * wout.transpose()).array() * z3.array() * (one2- z3).array();
		Mat w3_dw= z3.transpose()*w3_d;
		Mat w3_db= one.transpose()*w3_d;
		// w3.array() -= (lr * w3_dw.array());
		// b3.array() -= (lr * w3_db.array());
		w3-= (lr * w3_dw);
		b3-= (lr * w3_db);

		Mat w2_d = (w3_d * w3.transpose()).array() * z2.array() * (one2- z2).array();
		Mat w2_dw= z1.transpose()*w2_d;
		Mat w2_db= one.transpose()*w2_d;
		// w2.array() -= (lr * w2_dw.array());
		// b2.array() -= (lr * w2_db.array());
		w2 -= (lr * w2_dw);
		b2 -= (lr * w2_db);

		if (out_d.hasNaN() || out_dw.hasNaN() || out_db.hasNaN()) {
			std::cerr << "NaN detected in gradient calculation!" << std::endl;
		}
	};
	void show(){
		dump_all(b2);
	};
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
	Mat train_x(4,2);
	Mat train_y(4,1);
	train_x << 0,0,
						 0,1,
						 1,0,
						 1,1;
	train_y << 0,1,1,0;

	NN nn{2,8,8,1,0.1};
	// nn.show();

	// nn.test();

	for( int i =0;i <100;i++) {
		Mat out = nn.forward(train_x);
		nn.train(train_y);

		// std::cout << (double)get_correct_coutn(out,train_y)/4 << std::endl;
	};

	for(int i=0;i<4;i++)	{
		Mat x = train_x.row(i);
		Mat out = nn.forward(x);
		std :: cout << "in:" << x << "pred:" << out << std::endl;
	}
}
