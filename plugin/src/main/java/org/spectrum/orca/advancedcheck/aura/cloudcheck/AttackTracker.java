package org.spectrum.orca.advancedcheck.aura.cloudcheck;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class AttackTracker implements Listener {

    private final Map<Player, Long> playerAttackTimes = new ConcurrentHashMap<>();
    private final JavaPlugin plugin;
    private static AttackTracker instance;

    // 私有构造函数确保单例模式
    private AttackTracker(JavaPlugin plugin) {
        this.plugin = plugin;

        // 注册事件监听器
        Bukkit.getPluginManager().registerEvents(this, plugin);

        // 定时任务，每3秒清理一次数据
        Bukkit.getScheduler().runTaskTimer(plugin, () -> {
            long currentTime = System.currentTimeMillis();
            // 清理超出3秒的数据
            playerAttackTimes.entrySet().removeIf(entry ->
                    currentTime - entry.getValue() > 3000
            );
        }, 0L, 60L); // 每60tick执行一次（即每3秒执行一次）
    }

    // 获取单例实例
    public static AttackTracker getInstance(JavaPlugin plugin) {
        if (instance == null) {
            instance = new AttackTracker(plugin);
        }
        return instance;
    }

    @EventHandler
    public void onPlayerAttack(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player)) {
            return;
        }

        Player attacker = null;

        // 处理近战攻击
        if (event.getDamager() instanceof Player) {
            attacker = (Player) event.getDamager();
        }
        // 处理远程攻击（如箭、雪球等）
        else if (event.getDamager() instanceof Projectile) {
            Projectile projectile = (Projectile) event.getDamager();
            if (projectile.getShooter() instanceof Player) {
                attacker = (Player) projectile.getShooter();
            }
        }

        // 记录攻击者
        if (attacker != null) {
            playerAttackTimes.put(attacker, System.currentTimeMillis());
            //Bukkit.getLogger().info("[AttackTracker] 记录攻击: " + attacker.getName());
        }
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        // 玩家退出时清理数据
        playerAttackTimes.remove(event.getPlayer());
    }

    // 获取最近3秒内攻击过其他玩家的玩家列表
    public Set<Player> getPlayersAttacked() {
        long currentTime = System.currentTimeMillis();
        Set<Player> attackers = playerAttackTimes.entrySet().stream()
                .filter(entry -> currentTime - entry.getValue() <= 3000)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());

        //Bukkit.getLogger().info("[AttackTracker] 当前攻击者: " + attackers);
        return attackers;
    }
}
