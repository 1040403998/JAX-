


"""
代码示例 5.3 :
    使用递归进行纯函数的构造
"""

""" 循环 """
def summation(n):
    s = 0
    for i in range(n+1):
        s += i
    return s

print(summation(100))  # 5050

""" 递归 """
def summation(n):
    assert isinstance(n, int) and n >=0
    if n == 0:
        return 0
    else:      
        return n + summation(n-1)

print(summation(100))  # 5050

""" 尾递归 """
def summation(n, s=0):
    assert isinstance(n, int) and n >=0
    if n == 0:
        return s
    else:
        return summation(n-1, s+n)

print(summation(100))  # 5050


""" 递归深度测试

n=100
while n < 10000:    
    try: s = summation(n)
    except: break
    finally: n += 1
print(n)

"""

# 尾递归（求和列表）
def list_sum(num_list, s=0):
    if len(num_list) == 0:
        return s
    else:
        return list_sum(num_list[1:], s + num_list[0])

number_list = [i for i in range(101)]
print(list_sum(number_list))  # 5050