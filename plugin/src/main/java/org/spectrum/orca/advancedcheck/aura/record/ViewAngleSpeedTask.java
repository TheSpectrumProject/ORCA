package org.spectrum.orca.advancedcheck.aura.record;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.scheduler.BukkitRunnable;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Vector;

import java.util.HashMap;
import java.util.Map;

public class ViewAngleSpeedTask {

    private final JavaPlugin plugin;
    private final Map<Player, Vector> previousDirections = new HashMap<>();
    private final Map<Player, Double> previousAngles = new HashMap<>();  // 使用 double 存储角度
    private final Map<Player, Double> previousAngleSpeeds = new HashMap<>(); // 使用 double 存储角度速度（角度变化率）

    public ViewAngleSpeedTask(JavaPlugin plugin) {
        this.plugin = plugin;
    }

    // 启动定时任务
    public void start() {
        new BukkitRunnable() {
            @Override
            public void run() {
                for (Player player : Bukkit.getOnlinePlayers()) {
                    Vector currentDirection = player.getLocation().getDirection().normalize();
                    Vector previousDirection = previousDirections.get(player);

                    if (previousDirection != null) {
                        double dotProduct = currentDirection.dot(previousDirection);
                        dotProduct = Math.min(1.0, Math.max(-1.0, dotProduct));
                        double angle = Math.acos(dotProduct);
                        double angleInDegrees = Math.toDegrees(angle);

                        // 先获取之前保存的角度
                        Double previousAngle = previousAngles.get(player);
                        if (previousAngle != null) {
                            double angleSpeed = angleInDegrees - previousAngle;
                            previousAngleSpeeds.put(player, angleSpeed);
                        }

                        // 保存当前角度到previousAngles，供下次使用
                        previousAngles.put(player, angleInDegrees);
                    }

                    // 每次循环结束时更新previousDirections
                    previousDirections.put(player, currentDirection);
                }
            }
        }.runTaskTimer(plugin, 0L, 1L); // 确保以适当的周期运行
    }


    // 获取玩家的视角变化角度（返回 float 类型）
    public float getSpeed(Player player) {
        double speed = Math.abs(previousAngles.getOrDefault(player, 0.0));
        return (float) speed; // 返回 float 类型
    }

    // 获取玩家的视角变化角速度（返回 float 类型）
    public float getAcceleration(Player player) {
        double acc = Math.abs(previousAngleSpeeds.getOrDefault(player, 0.0));
        return (float) acc; // 返回 float 类型
    }
}
