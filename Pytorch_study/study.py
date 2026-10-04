print("Hello World!")

name = "hello"
print(name)

online = True

age = 20
price = 6.3
print(name, online, age, price)
print(name[0:2])
print(type(name), type(online), type(age), type(price))

new_price = float("3")
print(new_price, type(new_price))

array = [1,2,3,4,5,6]
print(len(array))
print(array[0], array[1])

my_dict = {"a":1, "b":2, "c":"hello"}
print(my_dict["a"],my_dict["c"])

if online:
    print("hello")
else:
    print("goodbye")

age = 20
if age < 35:
    print("young")
elif age < 65:
    print("middle")
else:
    print("old")

for x in array:
    print(x)

for (i, x) in enumerate(array):
    print(i,x)

counter = 100
while counter > 10:
    print(counter)
    counter -= 10

def sum_values(values):
    s = 0
    for v in values:
        s = s + v
    return s

print(sum_values(array))

import time

print("start")
time.sleep(2)
print("finish")

# 其他包用pip下载，网络不好更换镜像地址