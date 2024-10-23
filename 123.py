from itertools import count

name = "陈显力"
# year = 2000
# price = 19.99
# message = "我是：%s，我成立于：%d，我的价格是：%3.1f"%(name,year,price)
# print(f"我是{name}，我成立于{year}，我的价格是{price}")
def my_len(data):
    count=0
    for x in data:
        count=count+1
    return count



print(my_len(name))




