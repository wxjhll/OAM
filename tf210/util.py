import matplotlib.pyplot as plt

def show_loss(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plt.subplot(2, 1, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('MSE')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.savefig('loss_curve.png', dpi=100)
