import random

secret_number = random.randint(1, 100)
print("这是一个猜数字的游戏")

while True:
    try:
        print("请输入一个整数：")
        number = int(input())
        
        while number != secret_number:
            if number < secret_number:
                print("你猜的数字小了，请再试一次：")
            else:
                print("你猜的数字大了，请再试一次：")
            number = int(input())
        
        print("恭喜你，猜对了！")
        break
        
    except ValueError:
        print("请输入整数")