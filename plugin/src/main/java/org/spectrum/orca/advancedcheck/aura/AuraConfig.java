package org.spectrum.orca.advancedcheck.aura;

import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.plugin.Plugin;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;

public class AuraConfig {
    // 单例实例
    private static AuraConfig instance;
    // 插件主类引用
    private final Plugin plugin;
    // 配置相关变量
    private String apiEndpoint;
    private String apiKey;

    // 私有化构造方法
    private AuraConfig(Plugin plugin) {
        this.plugin = plugin;
        // 初始化时加载配置
        loadConfig();
    }

    /**
     * 获取单例实例（线程安全）
     * @param plugin 插件主类实例
     * @return 配置管理器单例
     */
    public static synchronized AuraConfig getInstance(Plugin plugin) {
        if (instance == null) {
            instance = new AuraConfig(plugin);
        }
        return instance;
    }

    /**
     * 加载/创建配置文件
     */
    private void loadConfig() {
        // 获取配置文件对象
        File configFile = new File(plugin.getDataFolder(), "AuraConfig.yml");

        // 如果配置文件不存在则创建
        if (!configFile.exists()) {
            plugin.saveResource("AuraConfig.yml", false); // false表示不覆盖现有文件
            plugin.getLogger().info("Created Default Aura Check Config");
        }

        // 加载配置
        FileConfiguration config = YamlConfiguration.loadConfiguration(configFile);

        // 读取配置值（带默认值）
        this.apiEndpoint = config.getString("api_endpoint", "http://127.0.0.1:5000/predict");
        this.apiKey = config.getString("api_key", "123456");

    }


    public String getApiEndpoint() {return apiEndpoint;}
    public String getApiKey() {return apiKey;}
}
