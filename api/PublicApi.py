from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import time
import redis
import logging
import json
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)

# Redis 配置
REDIS_HOST = 'localhost'  # 替换为你的 Redis 地址
REDIS_PORT = 6379
REDIS_DB = 0

# 初始化 Redis
redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=False  # 保持二进制解码以处理JSON数据
)

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path='A2-Flash.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 配置日志
logger = logging.getLogger('API_Logger')
logger.setLevel(logging.INFO)

# 创建文件处理器（记录到api.log）
file_handler = logging.FileHandler('api.log')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义日志格式
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# 添加处理器到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# API 密钥验证装饰器
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        api_key = request.headers.get('X-API-KEY')

        if not api_key:
            logger.info("未提供API密钥")
            return jsonify({'cheat_probability': 0}), 200

        # 检查 Redis 中是否存在该 API Key
        if not redis_client.exists(f"api_key:{api_key}"):
            logger.warning(f"无效的API密钥: {api_key}")
            return jsonify({'cheat_probability': 0}), 200

        # 检查请求频率
        rate_key = f"rate_limit:{api_key}"
        current_requests = redis_client.incr(rate_key)

        if current_requests == 1:
            redis_client.expire(rate_key, 60)  # 60 秒后重置计数

        max_requests = int(redis_client.get(f"api_key:{api_key}") or 60)

        if current_requests > max_requests:
            logger.warning(f"API密钥 {api_key} 请求频率超限")
            return jsonify({
                'cheat_probability': 0
            }), 429

        try:
            response = f(*args, **kwargs)
            return response
        except Exception as e:
            logger.error(f"处理请求出错: {str(e)}", exc_info=True)
            return jsonify({'cheat_probability': 0}), 200

    return decorated_function


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    try:
        data = request.get_json()
        if not data or 'time_series' not in data:
            logger.warning("无效的请求数据")
            return jsonify({'cheat_probability': 0}), 200

        time_series = list(map(float, data['time_series'].split(',')))
        if len(time_series) != 280:
            logger.warning(f"数据长度异常，期望280，实际收到{len(time_series)}")
            return jsonify({'cheat_probability': 0}), 200

        input_data = np.array(time_series, dtype=np.float32).reshape(1, 40, 7)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        cheat_prob = float(output[0][0])

        logger.info(f"预测成功 - 作弊概率: {cheat_prob:.4f}")
        return jsonify({'cheat_probability': cheat_prob}), 200

    except Exception as e:
        logger.error(f"预测过程中发生异常: {str(e)}", exc_info=True)
        return jsonify({'cheat_probability': 0}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
