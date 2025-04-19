package org.spectrum.orca.advancedcheck.aura.punish;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.spectrum.orca.advancedcheck.aura.cloudcheck.PunishPlayer;

import java.util.HashMap;
import java.util.Map;

public class ProbDecay {

    private static ProbDecay instance;
    private final Map<Player, Double> playerProbabilities = new HashMap<>();
    private int taskId;
    private final JavaPlugin plugin;

    private PunishPlayer punishPlayer;


    // 私有构造函数
    private ProbDecay(JavaPlugin plugin) {
        this.plugin = plugin;
        this.punishPlayer = PunishPlayer.getInstance(plugin);
    }

    // 获取单例实例
    public static ProbDecay getInstance(JavaPlugin plugin) {
        if (instance == null) {
            instance = new ProbDecay(plugin);
        }
        return instance;
    }

    public void start() {
        // 启动每秒执行的概率衰减任务
        taskId = Bukkit.getScheduler().scheduleSyncRepeatingTask(plugin, this::decayProbabilities, 20L, 20L);
    }

    public void stop() {
        // 取消任务
        Bukkit.getScheduler().cancelTask(taskId);
    }

    /**
     * 定时任务：每秒对所有玩家的概率进行衰减（×0.9）
     */
    private void decayProbabilities() {
        for (Player player : Bukkit.getOnlinePlayers()) {
            // 获取当前概率
            double currentProb = playerProbabilities.getOrDefault(player, 0.0);

            //检查是否封禁
            punishPlayer.punishBasedOnProbability(player,currentProb);

            // 应用衰减公式: S(t) = S(t) * 0.9
            double newProb = currentProb * 0.9;
            if (newProb<0){newProb=0;}

            // 更新概率
            playerProbabilities.put(player, newProb);

            // 调试输出（可选）
            //if (newProb > 0.1) {
            //    plugin.getLogger().info(player.getName() + " 概率衰减后: " + String.format("%.2f", newProb * 100) + "%");
            //}
        }
    }

    /**
     * 公开API：增加玩家的作弊概率
     * @param player 玩家对象
     * @param additionalProb 要增加的概率值 (0-1之间)
     */
    public void increaseProbability(Player player, double additionalProb) {

        // 获取当前概率
        double currentProb = playerProbabilities.getOrDefault(player, 0.0);

        // 增加概率: S(t) = S(t) + <新增概率> - 0.5
        double newProb = currentProb + additionalProb - 0.5;
        if (newProb<0){newProb=0;}

        // 更新概率
        playerProbabilities.put(player, newProb);

        // 调试输出（可选）
        //plugin.getLogger().info("为 " + player.getName() +
        //        " 增加概率: " + String.format("%.2f", additionalProb * 100) +
        //        "%，当前概率: " + String.format("%.2f", newProb * 100) + "%");
    }

    /**
     * 获取玩家的当前作弊概率
     * @param player 玩家对象
     * @return 当前作弊概率 (0-1之间)
     */
    public double getCheatProbability(Player player) {
        return playerProbabilities.getOrDefault(player, 0.0);
    }

    /**
     * 重置玩家的作弊概率
     * @param player 玩家对象
     */
    public void resetProbability(Player player) {
        playerProbabilities.put(player, 0.0);
        plugin.getLogger().info("已重置 " + player.getName() + " 的作弊概率");
    }
}
