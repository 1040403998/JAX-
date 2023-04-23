


"""
代码示例 4.3 :
    tensorflow 的 gradients 函数

    note: 需要使用 pip install tensorflow-cpu 安装环境
        或者可以采用以下代码进行换源, 加快库文件下载的速度:
        >> pip install tensorflow-cpu -i https://pypi.tuna.tsinghua.edu.cn/simple  

"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def f(x, y):
    return (x + y) ** 2

x = tf.compat.v1.Variable(1., dtype=tf.float32, name='x')
y = tf.compat.v1.Variable(2., dtype=tf.float32, name='y')
z = f(x, y)

df1, df2 = tf.gradients(z, [x, y], stop_gradients=[x,y])
df11 = tf.gradients(df1, x)
df12 = tf.gradients(df1, y)
df21 = tf.gradients(df2, x)
df22 = tf.gradients(df2, y)
df111 = tf.gradients(df11, x)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    """  第零阶  """
    print(sess.run(z))   # >> 9.0

    """  第一阶  """
    print(sess.run(df1))  # >> 6.0
    print(sess.run(df2))  # >> 6.0

    """  第二阶  """
    print(sess.run(df11)[0])  # >> 2.0
    print(sess.run(df12)[0])  # >> 2.0
    print(sess.run(df21)[0])  # >> 2.0
    print(sess.run(df22)[0])  # >> 2.0

    """  第三阶  """
    print(sess.run(df111)[0])  # >> 0.0

