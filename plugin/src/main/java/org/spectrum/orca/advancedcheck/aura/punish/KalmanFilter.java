package org.spectrum.orca.advancedcheck.aura.punish;

import java.util.LinkedList;

public class KalmanFilter {
    private double state;       // 初始状态
    private double covariance = 1.0; // 初始协方差
    private final double Q = 1e-5;   // 过程噪声
    private final double R = 0.1;    // 测量噪声
    private final LinkedList<Double> history = new LinkedList<>(); // 改用LinkedList

    public double filter(double measurement) {
        // 预测
        double predState = state;
        double predCov = covariance + Q;

        // 更新
        double K = predCov / (predCov + R);
        state = predState + K * (measurement - predState);
        covariance = (1 - K) * predCov;

        // 记录最新状态，并限制长度≤20
        history.add(state);
        if (history.size() > 20) {
            history.removeFirst(); // 移除最旧数据
        }

        return state;
    }
}
