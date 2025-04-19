package org.spectrum.orca.advancedcheck.aura.punish;

import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.ChatColor;
import org.bukkit.Bukkit;

import static org.spectrum.orca.advancedcheck.aura.punish.ActionBarUtils.sendActionBar;


public class DamageModifier implements Listener {

    private static DamageModifier instance;
    private final JavaPlugin plugin;
    private final ProbDecay probDecay;

    // 私有构造函数
    private DamageModifier(JavaPlugin plugin) {
        this.plugin = plugin;
        this.probDecay = ProbDecay.getInstance(plugin);
    }

    public void start() {
        plugin.getServer().getPluginManager().registerEvents(this, plugin);
    }
    public void stop(){
        EntityDamageByEntityEvent.getHandlerList().unregister(this);
    }

    // 获取单例
    public static DamageModifier getInstance(JavaPlugin plugin) {
        if (instance == null) {
            instance = new DamageModifier(plugin);
        }
        return instance;
    }

    @EventHandler
    public void onEntityDamage(EntityDamageByEntityEvent event) {
        if (!(event.getDamager() instanceof Player)) return;

        Player player = (Player) event.getDamager();
        double probability = probDecay.getCheatProbability(player);

        // 调试日志保持不变
        //plugin.getLogger().info("[DEBUG] 玩家 " + player.getName() + " 检测概率: " + probability);

        double originalDamage = event.getDamage();
        double modifiedDamage = originalDamage;
        String title = "";
        String subtitle = "";
        boolean shouldSendMessage = false;

        // 设置标题参数（单位：tick）
        int fadeIn = 0;   // 淡入时间 0秒
        int stay = 20;    // 保持时间 1秒
        int fadeOut = 0; // 淡出时间 0秒

        // 伤害调整逻辑
        if (probability > 3.7) {
            modifiedDamage *= 0.10;
            title = ChatColor.RED + "➤ 90% Damage Reduction Triggered";
            subtitle = ChatColor.YELLOW + "Suspicion Value: " + ChatColor.BOLD + "3.7+";
            shouldSendMessage = true;
        } else if (probability > 3.5) {
            modifiedDamage *= 0.30;
            title = ChatColor.GOLD + "➤ 70% Damage Reduction Triggered";
            subtitle = ChatColor.YELLOW + "Suspicion Range: 3.5-4.0";
            shouldSendMessage = true;
        } else if (probability > 3.0) {
            modifiedDamage *= 0.50;
            title = ChatColor.YELLOW + "➤ 50% Damage Reduction Triggered";
            subtitle = ChatColor.GRAY + "Suspicion Range: 3.0-3.5";
            shouldSendMessage = true;
        } else if (probability > 2.5) {
            modifiedDamage *= 0.80;
            title = ChatColor.WHITE + "➤ 30% Damage Reduction Triggered";
            subtitle = ChatColor.DARK_GRAY + "Suspicion Range: 2.5-3.0";
            shouldSendMessage = true;
        } else if (probability > 2) {
            modifiedDamage *= 0.80;
            title = ChatColor.WHITE + "➤ 20% Damage Reduction Triggered";
            subtitle = ChatColor.DARK_GRAY + "Suspicion Range: 2-2.5";
            shouldSendMessage = true;
        }


        if (modifiedDamage != originalDamage) {
            event.setDamage(modifiedDamage);

            // 发送屏幕标题
            if (shouldSendMessage) {
                plugin.getServer().getLogger().info("Player: " + player.getName() + ChatColor.stripColor(title) + ChatColor.stripColor(subtitle));
                for (Player onlinePlayer : Bukkit.getOnlinePlayers()) {
                    // 检查玩家是否有权限
                    if (onlinePlayer.hasPermission("orca.notify")) {
                        String prefix = ChatColor.RED + "[ORCA]";
                        onlinePlayer.sendMessage(prefix + "Player: " + ChatColor.YELLOW + player.getName() + title + subtitle);
                    }
                }
            }
        }
    }

}