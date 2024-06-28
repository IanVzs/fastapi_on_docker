import json
import time


def analyze(f):
    data = {}
    exec = json.loads(f[1])
    try:
        for i in exec["executes"]:
            execute_time = i["execute_time"]
            if i["status"] != "done":
                continue
            execute_datetime = time.strftime('%m-%d',time.localtime(i["execute_time"]/1000))
            if execute_datetime not in data:
                data[execute_datetime] = 1
            else:
                data[execute_datetime] += 1
    except:
        print(f"{data}")
    return data


def draw(data):
    import matplotlib.pyplot as plt

    # 提取日期和数值
    dates = sorted(data.keys())
    values = [data[date] for date in dates]
    # 创建柱状图
    plt.bar(dates, values)

    # 添加标题和标签
    plt.title('柱状图')
    plt.xlabel('日期')
    plt.ylabel('值')

    # 旋转 x 轴刻度标签，使其垂直显示
    plt.xticks(rotation=90)

    # 显示图形
    plt.show()