package org.spectrum.orca.advancedcheck.aura.cloudcheck;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.spectrum.orca.advancedcheck.aura.AuraConfig;
import org.spectrum.orca.advancedcheck.aura.record.RecordManager;
import org.bukkit.plugin.java.JavaPlugin;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import java.net.URL;

import org.json.JSONException;
import org.json.JSONObject;


public class PostRequest{

    private static String API_URL = "http://127.0.0.1:5000/predict";
    private static String API_KEY = "123456";

    private static String CACHED_API_URL;
    private static String CACHED_API_KEY;

    private final JavaPlugin plugin;
    private final RecordManager recordManager; // 添加这个成员变量

    private final AuraConfig auraConfig;

    public PostRequest(JavaPlugin plugin) {
        this.plugin = plugin;
        this.recordManager = RecordManager.getInstance(plugin); // 保存为成员变量
        this.auraConfig = AuraConfig.getInstance(plugin);
    }

    public double checkPlayer(String playerName){
        String csvData = recordManager.getRecordsAsCSV(playerName);

        // 调用 POST 请求方法将数据发送到 API
        //Bukkit.getLogger().info("data: " + csvData);4

        //去除末尾逗号
        csvData = csvData.replaceAll(",$", "");

        double cheatProbability = sendPostRequest(csvData);
        if (cheatProbability == -2){
            Bukkit.getLogger().info("API错误");
            return 0;}
        if (cheatProbability == -1){
            Bukkit.getLogger().info("数据清洗");
            return 0;}
        return cheatProbability;
    }

    private double sendPostRequest(String csvData) {

        //缓存API信息，？
        if(CACHED_API_URL == null){
            CACHED_API_URL = AuraConfig.getInstance(plugin).getApiEndpoint();
            CACHED_API_KEY = AuraConfig.getInstance(plugin).getApiKey();
        }
        API_URL = CACHED_API_URL;
        API_KEY = CACHED_API_KEY;

        try {
            // 1. 创建 URL 对象
            URL url = new URL(API_URL);

            // 2. 打开 HTTP 连接
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);

            // 3. 设置请求头为 JSON 格式
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("X-API-KEY", API_KEY);  // 添加API密钥到请求头

            // 4. 构建 JSON 请求体
            JSONObject jsonRequest = new JSONObject();
            jsonRequest.put("time_series", csvData); // 直接传 CSV 字符串

            // 5. 发送 JSON 数据
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = jsonRequest.toString().getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }

            // 6. 获取响应码
            int responseCode = connection.getResponseCode();
            //plugin.getServer().getLogger().info("POST Response Code: " + responseCode);

            // 7. 处理成功响应
            if (responseCode == HttpURLConnection.HTTP_OK) {
                try (BufferedReader in = new BufferedReader(
                        new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder response = new StringBuilder();
                    String inputLine;
                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }

                    // 8. 解析 JSON 获取概率值
                    JSONObject jsonResponse = new JSONObject(response.toString());
                    double cheatProbability = jsonResponse.getDouble("cheat_probability");

                    //plugin.getServer().getLogger().info("API返回的欺诈概率值: " + cheatProbability);
                    return cheatProbability;
                } catch (JSONException e) {
                    plugin.getServer().getLogger().severe("JSON解析失败: " + e.getMessage());
                    return -2;
                }
            } else {
                // 9. 处理错误响应
                try (BufferedReader in = new BufferedReader(
                        new InputStreamReader(connection.getErrorStream(), StandardCharsets.UTF_8))) {
                    StringBuilder errorResponse = new StringBuilder();
                    String inputLine;
                    while ((inputLine = in.readLine()) != null) {
                        errorResponse.append(inputLine);
                    }
                    plugin.getServer().getLogger().severe("POST请求失败，错误响应: " + errorResponse);
                }
                return -2;
            }
        } catch (IOException e) {
            plugin.getServer().getLogger().severe("Error sending POST request: " + e.getMessage());
            e.printStackTrace();
            return -2;
        }
    }


}
