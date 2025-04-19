package org.spectrum.orca.advancedcheck.aura.record;

import org.bukkit.Bukkit;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

public class RecordManager {
    private static RecordManager instance;
    private final JavaPlugin plugin;
    private final ConcurrentHashMap<String, Queue<Record>> recordDatabase = new ConcurrentHashMap<>();

    // 常量
    private static final int MAX_RECORDS_PER_PLAYER = 50;
    private static final int MAX_RECORDS_TO_RETURN = 40;

    // 私有构造函数
    private RecordManager(JavaPlugin plugin) {
        this.plugin = Objects.requireNonNull(plugin, "Plugin cannot be null!");
    }

    // 获取单例
    public static synchronized RecordManager getInstance(JavaPlugin plugin) {
        if (instance == null) {
            instance = new RecordManager(plugin);
        }
        return instance;
    }

    // 保存记录
    public void saveToDb(String playerName, float sinPitch, float sinYaw, float cosYaw,
                         float speed, float acc, float angle, String status) {
        if (playerName == null || status == null) return;

        Queue<Record> playerRecords = recordDatabase.computeIfAbsent(
                playerName, k -> new ConcurrentLinkedQueue<>()
        );

        if (playerRecords.size() >= MAX_RECORDS_PER_PLAYER) {
            playerRecords.poll(); // 自动移除最旧记录
        }

        playerRecords.offer(new Record(
                playerName, sinPitch, sinYaw, cosYaw, speed, acc, angle, status
        ));
    }

    // 获取最近记录
    public List<Record> getRecentRecords(String playerName) {
        Queue<Record> playerRecords = recordDatabase.get(playerName);
        if (playerRecords == null || playerRecords.size() < MAX_RECORDS_TO_RETURN) {
            return new ArrayList<>();
        }
        return new ArrayList<>(playerRecords).subList(
                Math.max(playerRecords.size() - MAX_RECORDS_TO_RETURN, 0),
                playerRecords.size()
        );
    }

    // 生成 CSV
    public String getRecordsAsCSV(String playerName) {
        List<Record> recentRecords = getRecentRecords(playerName);
        if (recentRecords.isEmpty()) {
            Bukkit.getLogger().warning("No records found for player: " + playerName);
            return "";
        }

        StringBuilder csv = new StringBuilder();
        for (Record record : recentRecords) {
            csv.append(record.sinPitch).append(",")
                    .append(record.sinYaw).append(",")
                    .append(record.cosYaw).append(",")
                    .append(record.speed).append(",")
                    .append(record.acc).append(",")
                    .append(record.angle).append(",")
                    .append(record.status).append(",");
        }
        return csv.toString();
    }

    // 关闭清理
    public void shutdown() {
        plugin.getLogger().info("Shutdown: Total players with records: " + recordDatabase.size());
        recordDatabase.clear();
    }

    // 内部记录类
    private static class Record {
        final String playerName;
        final float sinPitch, sinYaw, cosYaw;
        final float speed, acc, angle;
        final String status;

        Record(String playerName, float sinPitch, float sinYaw, float cosYaw,
               float speed, float acc, float angle, String status) {
            this.playerName = playerName;
            this.sinPitch = sinPitch;
            this.sinYaw = sinYaw;
            this.cosYaw = cosYaw;
            this.speed = speed;
            this.acc = acc;
            this.angle = angle;
            this.status = status;
        }
    }
}
