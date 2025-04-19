package org.spectrum.orca.advancedcheck.aura.cloudcheck;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

public class PunishPlayer {
    private static PunishPlayer instance;
    private final JavaPlugin plugin;

    // 私有构造函数
    private PunishPlayer(JavaPlugin plugin) {
        this.plugin = plugin;
    }

    // 获取单例实例
    public static PunishPlayer getInstance(JavaPlugin plugin) {
        if (instance == null) {
            instance = new PunishPlayer(plugin);
        }
        return instance;
    }

    /**
     * 根据概率惩罚玩家
     * @param player 要惩罚的玩家
     * @param prob 当前检测到的作弊概率
     */
    public void punishBasedOnProbability(Player player, double prob) {
        // 确保在主线程执行玩家操作
        Bukkit.getScheduler().runTask(plugin, () -> {
            if (prob > 4.0) {
                // 踢出玩家
                //player.kickPlayer("§c [ORCA] Unfair Advantage");
                //plugin.getLogger().info("已踢出玩家 " + player.getName() + "，检测概率: " + prob);

                // 可以在这里添加其他惩罚措施
                // 如封禁、记录日志等
            } else if (prob > 2.8) {
                // 中等概率警告
                //player.sendMessage("§c [ORCA] 减伤30%");
                //plugin.getLogger().info("警告玩家 " + player.getName() + "，检测概率: " + prob);
            }
            // 低概率不采取行动
        });
    }

    /**
     * 重置单例实例（用于重新加载插件时）
     */
    public static void resetInstance() {
        instance = null;
    }
}
