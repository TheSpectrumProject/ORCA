package org.spectrum.orca.advancedcheck.aura.punish;

import org.bukkit.plugin.java.JavaPlugin;

public class PunishMain {

    private final JavaPlugin plugin;
    private ProbDecay probDecay;

    private DamageModifier damageModifier;

    public PunishMain(JavaPlugin plugin) {
        this.plugin = plugin;
        this.probDecay = ProbDecay.getInstance(plugin);
        this.damageModifier = DamageModifier.getInstance(plugin);
    }

    /**
     * 启动惩罚系统
     */
    public void start() {
        // 启动概率衰减系统,包含定时惩罚
        probDecay.start();
        damageModifier.start();

        // 可以在这里初始化其他子系统
        //plugin.getLogger().info("作弊惩罚系统已启动");
    }

    /**
     * 禁用惩罚系统
     */
    public void stop() {
        // 停止概率衰减系统
        probDecay.stop();
        damageModifier.stop();

        // 可以在这里清理其他子系统
        //plugin.getLogger().info("作弊惩罚系统已禁用");
    }
}
