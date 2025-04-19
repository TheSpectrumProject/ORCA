package org.spectrum.orca.advancedcheck.aura.record;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

public class RecordMain {

    private JavaPlugin plugin;  // 确保 plugin 变量类型为 JavaPlugin
    private ViewAngleSpeedTask viewAngleSpeedTask;
    private GetNearAngleTask getNearAngleTask;
    private RecordManager recordManager;

    public RecordMain(JavaPlugin plugin) {
        this.plugin = plugin;  // 在构造函数中传入 JavaPlugin 主类实例
        this.viewAngleSpeedTask = new ViewAngleSpeedTask(this.plugin);
        this.getNearAngleTask = new GetNearAngleTask();
    }

    public void start() {
        // 注册事件监听器
        plugin.getServer().getPluginManager().registerEvents(new ClickListener(this), plugin);//传入auraMain以便调用recordSingle
        viewAngleSpeedTask.start();

        recordManager = RecordManager.getInstance(plugin);//获取单例

        // 启动定时任务
        Bukkit.getScheduler().runTaskTimer(plugin, () -> {
            for (Player player : Bukkit.getOnlinePlayers()) {
                recordSingleView(player, "0");
            }
        }, 0L, 1L);
    }
    public void stop() {
        recordManager.shutdown();
    }

    public boolean recordSingleView(Player player, String click) {
        if ("bot".equals(player.getName())) {
            return false;
        }
        if ("admin".equals(player.getName())) {
            return false;
        }

        Location playerLocation = player.getLocation();

        float sinPitch = (float) Math.sin(Math.toRadians(playerLocation.getPitch()));
        float sinYaw = (float) Math.sin(Math.toRadians(playerLocation.getYaw()));
        float cosYaw = (float) Math.cos(Math.toRadians(playerLocation.getYaw()));

        sinPitch = (sinPitch+1)/2;
        sinYaw = (sinYaw+1)/2;
        cosYaw = (cosYaw+1)/2;


        float speed = viewAngleSpeedTask.getSpeed(player);
        float acc = viewAngleSpeedTask.getAcceleration(player);
        float angle = getNearAngleTask.getAngle(player);

        speed = speed/10;
        acc = acc/10;

        angle = (float) Math.cos(angle);
        angle = (angle+1)/2;

        recordManager.saveToDb(player.getName(), sinPitch, sinYaw, cosYaw, speed, acc, angle, click);
        return true;
    }
}
