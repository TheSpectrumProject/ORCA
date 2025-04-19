from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import time  # 新增导入time模块

app = Flask(__name__)

# 加载量化后的TFLite模型
interpreter = tf.lite.Interpreter(model_path='A2-Flash.tflite')
interpreter.allocate_tensors() # 分配张量内存

# 获取输入/输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 解析输入数据
        data = request.get_json()
        if 'time_series' not in data:
            return jsonify({'error': 'Missing "time_series" key'}), 400

        # 转换为浮点数组（混合精度输入为float32）
        time_series = list(map(float, data['time_series'].split(',')))
        if len(time_series) != 280:
            return jsonify({'error': 'Need exactly 280 values'}), 400

        # 预处理（与原始模型相同）
        input_data = np.array(time_series, dtype=np.float32).reshape(1, 40, 7)

        # 设置输入张量（注意：输入类型应为float32）
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 执行推理并记录时间
        start_time = time.time()  # 记录推理开始时间
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # 计算耗时，转换为毫秒

        # 获取输出（自动保持浮点）
        output = interpreter.get_tensor(output_details[0]['index'])
        cheat_prob = float(output[0][0]) # 假设输出是二维概率值

        print(inference_time)

        return jsonify({
            'cheat_probability': cheat_prob
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
