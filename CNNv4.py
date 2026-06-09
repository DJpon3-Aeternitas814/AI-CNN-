import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 仅显示错误日志
import numpy as np
import pickle

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(2)
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
        self.dropout = tf.keras.layers.Dropout(0.5)  # 添加50%概率的神经元丢弃

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)  # 仅在训练时启用
        return self.dense2(x)

def load_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='bytes')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 训练文件列表（转换为绝对路径）
    train_files = [
        os.path.join(script_dir, f'data/cifar-10-batches-py/data_batch_{i}') 
        for i in range(1, 6)
    ]

    # 测试文件（转换为绝对路径）
    test_file = os.path.join(script_dir, 'data/cifar-10-batches-py/test_batch')
    
    # 加载训练数据
    train_data = []
    train_labels = []
    for f in train_files:
        batch = unpickle(f)
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    # 加载测试数据
    test_batch = unpickle(test_file)
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # 数据预处理
    def preprocess(data):
        data = data.reshape(-1, 3, 32, 32).transpose(0,2,3,1)
        return data.astype(np.float32) / 255.0
    
    x_train = preprocess(np.concatenate(train_data))
    x_test = preprocess(test_data)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 创建数据集
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_ds, test_ds

def train_epoch(model, optimizer, loss_fn, train_ds, train_acc_metric):
    for x_batch, y_batch in train_ds:
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss = loss_fn(y_batch, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(y_batch, preds)
    return loss, train_acc_metric.result()

def test_epoch(model, loss_fn, test_ds, test_acc_metric):
    for x_batch, y_batch in test_ds:
        preds = model(x_batch, training=False)
        loss = loss_fn(y_batch, preds)
        test_acc_metric.update_state(y_batch, preds)
    return loss, test_acc_metric.result()

def main():

    # 获取当前脚本路径，注意PBS系统中运行程序一定要绝对路径！！！
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载数据
    train_ds, test_ds = load_data()
    
    # 准备数据集
    BATCH_SIZE = 128
    train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 初始化模型和组件
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    modelsaveweight = os.path.join(script_dir, "best_model.h5")
    # 训练循环
    best_acc = 0.0
    EPOCHS = 10
    for epoch in range(EPOCHS):
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, optimizer, loss_fn, train_ds, train_acc_metric
        )
        # 测试阶段
        test_loss, test_acc = test_epoch(
            model, loss_fn, test_ds, test_acc_metric
        )
        
        # 重置指标状态
        train_acc_metric.reset_state()
        test_acc_metric.reset_state()
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            model.save_weights(modelsaveweight)
        
        print(f"第 {epoch+1}/{EPOCHS} 轮训练")
        print(f"训练损失: {train_loss:.4f}  准确率: {train_acc:.4f}")
        print(f"测试损失: {test_loss:.4f}  准确率: {test_acc:.4f}\n")
    
    # 加载最佳权重
    model.load_weights(modelsaveweight)
    
    # 最终评估
    test_acc_metric.reset_state()
    _, test_acc = test_epoch(model, loss_fn, test_ds, test_acc_metric)
    print(f"最终测试准确率: {test_acc:.4f}")
    
    # 保存完整模型
    modelsavepath = os.path.join(script_dir, "model/cifar10_cnn_tf.keras")
    model.save(modelsavepath) 

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    main()