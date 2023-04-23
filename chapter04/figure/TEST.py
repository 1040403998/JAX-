


# """
# 代码示例 4.2.2 :
#     使用多项式 Pn(x) 拟合数据 (L1正则化)
# """

# import jax
# import jax.numpy as jnp
# from jax import jacfwd, jacrev

# n = 20  # parameters' number
# N = 10   # sample's number

# n-=1
# params = jnp.ones(n+1)

# # 定义用于拟合的多项式
# def Pn(x, params):
#     assert len(params) == n+1
#     res = 0.
#     for i in range(n+1):
#         res += params[i] * x ** (i)
#     return res    

# # 损失函数
# def loss(params, x_sample, y_sample):
#     return jnp.mean((Pn(x_sample, params) - y_sample) ** 2)

# def hessian(f): return jacfwd(jacrev(f))

# key = jax.random.PRNGKey(1)

# hf = hessian(loss)
# x_array = jnp.arange(N)
# y_array = jax.random.gamma(key, a = 0.5, shape=(N,)) * 10
# H = hf(params, x_array, y_array)
# H /= jnp.mean(H)
# print(H)
# H = (H + H.T) / 2
# eigval = jnp.linalg.eigvals(H)
# print("eigal = ", eigval)

# real = jnp.real(eigval)
# imag = jnp.imag(eigval)
# eigval = jnp.abs(eigval)
# print(real)
# print(imag)
# print(jnp.linalg.det(H))
# import matplotlib.pyplot as plt

# eps = 1E-8
# # plt.scatter(jnp.arange(len(eigval)), jnp.log(eigval), label = "abs")

# plt.scatter(jnp.arange(len(eigval)), jnp.log(eigval+eps), label = "abs", c= "r", s = 20)
# plt.scatter(jnp.arange(len(eigval)), jnp.log(jnp.abs(real)+eps), label = "real", c="y", s = 15)
# plt.scatter(jnp.arange(len(eigval)), jnp.log(jnp.abs(imag)+eps), label = "imag", c="g", s = 15)
# plt.legend(loc = "upper right")
# plt.savefig("eigval.png")

# print(sum(jnp.log(jnp.abs(real)+eps) > 35))



import matplotlib.pyplot as plt
import numpy as np
import sys

__all__ = ['poly2_estimator']

class poly2_estimator(object):
    def __init__(self,draw = True,  **kwargs):
        self.a1 = 1.
        self.a0 = 0.
        self.x_mean = None
        self.y_mean = None
        self.x_values = None
        self.y_values = None
        self.var = None



        self.draw_ = draw
        self.draw_enabled = False
        if draw:
            self.dot_size = 40

            self.x_label = "x channel"
            self.y_label = "y channel"
            self.title = "title"
            if "title" in kwargs:
                self.title = kwargs['title']
            if "x_label" in kwargs:
                self.x_label = kwargs['x_label']
            if "y_label" in kwargs:
                self.y_label = kwargs['y_label']

            self.plot_kwarg = {"color": "red" , "linewidth" : 2, "label" : "fitting line"}
            self.scatter_kwarg = {"color": "green", "marker": "o", "label": "Data point"}
            self.fig, self.ax = plt.subplots()

            # the range of x while drawing
            self.x_range_min = 0.2
            self.x_range_max = 0.9


    def __call__(self, x):
        return self.function(x)

    def function(self, x):
        return self.a1 * x + self.a0

    def fit(self, x_values, y_values):
        self.x_values = np.array(x_values,dtype=float)
        self.y_values = np.array(y_values,dtype=float)
        self.x_mean = np.average(x_values)
        self.y_mean = np.average(y_values)

        x_square = np.sum((x_values-self.x_mean)**2)
        self.a1 = np.sum((self.y_values - self.y_mean) * (self.x_values-self.x_mean)) / x_square
        self.a0 = (self.y_mean * np.sum(self.x_values**2) - np.sum(self.x_values * self.y_values)*self.x_mean)/ (x_square)

        print("y_pred = a1 * x + a0, a1 = {:.5f}, a0 = {:.5f}".format(self.a1, self.a0))

        if self.draw_:
            self.scatter_kwarg['s'] = self.dot_size * np.ones_like(x_values)
            self.draw_enabled = True
            if self.x_range_min > min(self.x_values):
                self.x_range_min = min(self.x_values)
            if self.x_range_max < max(self.x_values):
                self.x_range_max = max(self.x_values)

    def get_parameters(self, pnt = True):
        self.xy_mean = np.mean(self.x_values * self.y_values)
        self.x_xi_norm2_mean = np.sqrt(np.mean((self.x_values-self.x_mean)**2))
        self.y_yi_norm2_mean = np.sqrt(np.mean((self.y_values-self.y_mean)**2))
        self.r = (self.xy_mean - self.x_mean*self.y_mean)/ (self.x_xi_norm2_mean * self.y_yi_norm2_mean)
        print("r =", self.r)
        return self.x_xi_norm2_mean


    def draw(self, render = True, dots_num = 6):
        if not self.draw_enabled:
            print("Haven't input your sample data yet!!")
            return None

        x_plot = np.linspace(self.x_range_min, self.x_range_max, dots_num)
        y_plot = self.function(x_plot)


        self.ax.scatter(self.x_values, self.y_values, **self.scatter_kwarg)
        self.ax.plot(x_plot, y_plot, **self.plot_kwarg)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.grid('-')
        plt.legend(loc = "lower right")
        if render:
            plt.show()
        return self.a1 , self.a0


def test():
    # generate sample
    a = 1
    b = 2
    print(a,b)
    x_samples = np.arange(1,6,0.05)
    noise = np.random.randn(x_samples.shape[0]) * 0.2
    y_samples = a * x_samples + b + noise

    # fitting the data
    estimator = poly2_estimator(draw=False)
    estimator.fit(x_samples, y_samples)
    estimator.draw()

if __name__ == "__main__":
    x_num = np.array([1,2,3,4,6,7,8,])
    y_num = np.array([50.442,50.362,50.280,50.205,50.048,49.968,49.885])
    kwarg = {'title': None, "y_label": "distance(mm)", "x_label": 'Index'}

    estimator = poly2_estimator(draw=True, **kwarg)
    estimator.fit(x_num, y_num)
    estimator.draw()
    estimator.get_parameters()
    # sys.exit(test())
    # print(3E8 / 9370000)