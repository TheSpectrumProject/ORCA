package org.spectrum.orca.advancedcheck.aura.cloudcheck;

import org.bukkit.event.HandlerList;
import org.bukkit.plugin.java.JavaPlugin;

public class CloudMain {
    private final JavaPlugin plugin;
    private CheckPlayer checkPlayer;
    private AttackTracker attackTracker;

    public CloudMain(JavaPlugin plugin) {
        this.plugin = plugin;
        this.checkPlayer = new CheckPlayer(plugin);  // 使用无参构造函数
    }


    public void start() {
        AttackTracker.getInstance(plugin);
        checkPlayer.start();
        //plugin.getLogger().info("Cloud Aura Check Started");
    }

    public void stop() {
        if (attackTracker != null) {
            HandlerList.unregisterAll(attackTracker);
        }
        if (checkPlayer != null) {
            //checkPlayer.stop();
        }
        //plugin.getLogger().info("Cloud Aura Check Stopped");
    }
}
