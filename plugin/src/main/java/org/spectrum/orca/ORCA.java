package org.spectrum.orca;

import org.bukkit.plugin.java.JavaPlugin;
import org.spectrum.orca.advancedcheck.aura.AuraMain;

public class ORCA extends JavaPlugin {

    private AuraMain auraMain;

    @Override
    public void onEnable() {
        try {
            auraMain = new AuraMain(this);

            auraMain.start();

            getLogger().info("ORCA AntiCheat Loaded");
        } catch (Exception e) {
            getLogger().severe("ORCA AntiCheat Load ERROR: " + e.getMessage());
            e.printStackTrace();
            getServer().getPluginManager().disablePlugin(this);
        }
    }

    @Override
    public void onDisable() {
        if (auraMain != null) {
            auraMain.stop();
        }
        getLogger().info("ORCA 已关闭。");
    }
}
