package org.spectrum.orca.advancedcheck.aura;

import org.bukkit.plugin.java.JavaPlugin;
import org.spectrum.orca.advancedcheck.aura.cloudcheck.CloudMain;
import org.spectrum.orca.advancedcheck.aura.punish.PunishMain;
import org.spectrum.orca.advancedcheck.aura.record.RecordMain;

public class AuraMain {  // 移除了extends JavaPlugin，改为final类

    private final JavaPlugin plugin;
    private RecordMain recordMain;
    private CloudMain cloudMain;
    private PunishMain punishMain;

    // 通过构造函数传入JavaPlugin实例
    public AuraMain(JavaPlugin plugin) {
        this.plugin = plugin;
    }

    public void start() {
        try {
            recordMain = new RecordMain(plugin);
            cloudMain = new CloudMain(plugin);
            punishMain = new PunishMain(plugin);

            recordMain.start();
            cloudMain.start();
            punishMain.start();

            plugin.getLogger().info("Aura Check Loaded");
        } catch (Exception e) {
            plugin.getLogger().severe("Aura Check Load ERROR: " + e.getMessage());
            e.printStackTrace();
            plugin.getServer().getPluginManager().disablePlugin(plugin);
        }
    }

    public void stop() {
        if (recordMain != null) {
            recordMain.stop();
        }
        if (cloudMain != null) {
            cloudMain.stop();
        }
        if (punishMain != null) {
            punishMain.stop();
        }
        plugin.getLogger().info("Aura Check Stopped");
    }
}
