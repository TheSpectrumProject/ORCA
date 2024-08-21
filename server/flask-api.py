from flask import request, jsonify,Flask, render_template
import pandas as pd
import predict

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def save_csv():
    data = request.json  # 获取 JSON 数据

    if not data or 'data' not in data:
        return jsonify({'result': 'ERROR:Post request does not contain JSON dataframe!'})

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data['data'])

    # 指定 CSV 文件路径
    csv_path = 'data_output.csv'

    # 保存 DataFrame 为 CSV 文件
    df.to_csv(csv_path, index=False)
    result = predict.main("data_output.csv")
    print(result)
    return jsonify({'result': f'{result}'})



@app.route('/predict', methods=['GET'])
def get_index():
    return render_template("index.html")



@app.route("/")
def root():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True,port=5000)
