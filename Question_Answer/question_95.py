import numpy as np

np.random.seed(0)

class NN:
    #初期化（__init__メソッド）
    def __init__(self, ind=2, w1=64, w2=64, outd=1, lr=0.1):
        #ind=2: 入力層のユニット数（XORの場合、入力は2つ）
        #w=64: 中間層（隠れ層）のユニット数（64）
        #outd=1: 出力層のユニット数（XORの出力は1つ）
        #lr=0.1: 学習率
        #初期化の際に、各層の重み（w1, wout）とバイアス（b1, bout）をランダムに設定しています。
        ### np.random.normal(0, 1, [ind, w])
        ### **正規分布（ガウス分布）**に従ったランダムな数値を生成する関数
        ### np.random.normal(loc, scale, size)
        ###  1.  loc (0): 正規分布の**平均（μ）**です。この場合、0が指定されているので、平均が0の正規分布に従ってランダムな値が生成されます。
        ###  2.  scale (1): 正規分布の**標準偏差（σ）**です。この場合、1が指定されているので、標準偏差が1の正規分布に従ってランダムな値が生成されます。
        ###  3.  size ([ind, w]): 出力する配列の形状です。この場合、[ind, w]が指定されているので、ind行とw列の2次元配列が生成されます。
        
        # 入力層から第一中間層
        self.w1 = np.random.normal(0, 1, [ind, w1])
        self.b1 = np.random.normal(0, 1, [w1])
        # 第一中間層から第二中間層
        self.w2 = np.random.normal(0,1,[w1,w2])
        self.b2 = np.random.normal(0,1,[w2])
        # 第二中間層から出力層
        self.wout = np.random.normal(0, 1, [w2, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr
    #順伝播（forwardメソッド）
    def forward(self, x):
        #入力 x が与えられると、順に計算が進みます。
        #1.  入力層から隠れ層: z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # 入力 x と重み w1 の積を計算し、バイアス b1 を足し、シグモイド関数を適用して隠れ層の出力 z2 を求めます。
        #2.  隠れ層から出力層: out = sigmoid(np.dot(self.z2, self.wout) + self.bout)
        # 隠れ層の出力 z2 と重み wout の積を計算し、バイアス bout を足し、シグモイド関数を適用して最終的な出力 out を求めます。
        # シグモンと関数とは？
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out
    #学習（trainメソッド）
    def train(self, x, t):
        # backpropagation output layer
        #逆伝播（バックプロパゲーション） で、誤差を計算し、それに基づいて重みやバイアスを更新します。
        #1.  出力層の誤差:
        # En = (self.out - t) * self.out * (1 - self.out) で、出力層の誤差 En を計算します。
        # これは、シグモイド関数の微分を利用した誤差の計算です。
        #2.  出力層の勾配:
        # grad_wout と grad_bout を計算し、重み wout とバイアス bout を更新します。
        #3.  隠れ層の誤差:
        # •  grad_u1 = np.dot(En, self.wout.T) * self.z2 * (1 - self.z2) で隠れ層の誤差 grad_u1 を計算します。
        #4.  隠れ層の勾配:
        # •  grad_w1 と grad_b1 を計算し、重み w1 とバイアス b1 を更新します。
        En = (self.out - t) * self.out * (1 - self.out)
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout
        
				# backpropagation inter layer
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        
        # backpropagation inter layer
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# train
def train_nn(nn, train_x, train_t, iteration_N=5000):
    for i in range(5000):
        # feed-forward data
        nn.forward(train_x)
        #print("ite>>", i, 'y >>', nn.forward(train_x))
        # update parameters
        nn.train(train_x, train_t)

    return nn

# test
def test_nn(nn, test_x, test_t):
    for j in range(len(test_x)):
        x = train_x[j]
        t = train_t[j]
        print("in:", x, "pred:", nn.forward(x))


nn = NN()

# train data
train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)

# train label data
train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

# prepare neural network
nn = NN()

# train
nn = train_nn(nn, train_x, train_t, iteration_N=5000)

# test
test_nn(nn, train_x, train_t)