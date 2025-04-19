package org.spectrum.orca.advancedcheck.aura.cloudcheck;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.event.Listener;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.ChatColor;

import org.spectrum.orca.advancedcheck.aura.punish.ProbDecay;


import java.util.Set;

public class CheckPlayer implements Listener {

    private final JavaPlugin plugin;
    private final PostRequest postRequest;

    private final ProbDecay probDecay; // 添加这个字段


    public CheckPlayer(JavaPlugin plugin) {
        this.plugin = plugin;
        this.postRequest = new PostRequest(plugin);
        this.probDecay = ProbDecay.getInstance(plugin);
    }



    public void start() {
        // 定时任务：每秒检查一次攻击的玩家
        Bukkit.getScheduler().runTaskTimer(plugin, () -> {

            AttackTracker tracker = AttackTracker.getInstance(plugin);

            Set<Player> attackedPlayers = tracker.getPlayersAttacked();
            // 如果有攻击的玩家，执行特定操作
            if (!attackedPlayers.isEmpty()) {
                for (Player player : attackedPlayers) {
                    handlePlayerAttack(player);
                }
            }
        }, 0L, 20L); // 每20tick执行一次（即每1秒执行一次）
    }

    private void handlePlayerAttack(Player player) {
        // 在主线程获取需要的数据（因为Player对象不是线程安全的）
        String playerName = player.getName();

        // 异步执行检测逻辑
        Bukkit.getScheduler().runTaskAsynchronously(plugin, () -> {
            // 在异步线程中执行耗时操作
            double cheatProbability = postRequest.checkPlayer(playerName);

            // 异步操作完成后切换回主线程向玩家发送消息
            Bukkit.getScheduler().runTask(plugin, () -> {
                if (player != null && player.isOnline()) {
                    // 格式化概率为百分比，保留2位小数
                    //String percentage = String.format("%.2f%%", cheatProbability * 100);

                    // 根据概率值给出不同颜色的消息
                    //if (cheatProbability > 0.9) {
                    //    player.sendMessage(ChatColor.RED + "⚠ 检测到高作弊概率: " + percentage);
                    //} else if (cheatProbability > 0.7) {
                    //    player.sendMessage(ChatColor.YELLOW + "⚠ 检测到可疑行为: " + percentage);
                    //} else {
                    //    player.sendMessage(ChatColor.GREEN + "✓ 检测结果正常: " + percentage);
                    //}

                    // 可选：记录到控制台
                    //plugin.getLogger().info("玩家 " + playerName + " 作弊概率检测结果: " + percentage);
                    if (cheatProbability >0){
                        updatePro(player,cheatProbability);
                    }
                }
            });
        });

    }

    private void updatePro(Player player, Double pro){
        probDecay.increaseProbability(player, pro);
    }



}
