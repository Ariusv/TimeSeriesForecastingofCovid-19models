import matplotlib.pyplot as plt
def plot(series1,label1, series2, label2, title):
    plt.figure(figsize=(10, 6))
    plt.plot(series1, color='blue', label=label1)
    plt.plot(series2, color='red', label=label2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.show()