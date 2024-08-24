package org.spectrum.orca;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Vector;

import java.io.*;
import java.util.*;

public class ORCA extends JavaPlugin implements CommandExecutor, Listener {

    private final Map<String, DataCollector> collectors = new HashMap<>();
    private final Map<String, List<Integer>> attackTicks = new HashMap<>();
    private final Map<String, Boolean> isAttacking = new HashMap<>();
    private int tickCounter = 0;

    @Override
    public void onEnable() {
        // 确保数据文件夹存在
        if (!getDataFolder().exists()) {
            getDataFolder().mkdirs();
        }

        this.getCommand("orca").setExecutor(this);
        Bukkit.getPluginManager().registerEvents(this, this);
        Bukkit.getScheduler().runTaskTimer(this, this::onTick, 0, 1);
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length != 3) {
            return false;
        }

        String subCommand = args[0];
        String playerName = args[1];
        String action = args[2];
        Player player = Bukkit.getPlayer(playerName);

        if (player == null) {
            sender.sendMessage("玩家未找到。");
            return true;
        }

        switch (subCommand) {
            case "collectdata":
                if ("start".equals(action)) {
                    startCollectingData(playerName);
                    sender.sendMessage("开始收集 " + playerName + " 的数据.");
                } else if ("stop".equals(action)) {
                    stopCollectingData(playerName);
                    sender.sendMessage("停止收集 " + playerName + " 的数据.");
                } else {
                    return false;
                }
                break;
            default:
                return false;
        }
        return true;
    }

    private void startCollectingData(String playerName) {
        collectors.put(playerName, new DataCollector(playerName));
        attackTicks.put(playerName, new ArrayList<>());
        isAttacking.put(playerName, false);
    }

    private void stopCollectingData(String playerName) {
        DataCollector collector = collectors.remove(playerName);
        attackTicks.remove(playerName);
        isAttacking.remove(playerName);
        if (collector != null) {
            collector.saveData();
        }
    }

    private void onTick() {
        tickCounter++;
        for (Map.Entry<String, DataCollector> entry : collectors.entrySet()) {
            String playerName = entry.getKey();
            DataCollector collector = entry.getValue();
            boolean attacking = isAttacking.getOrDefault(playerName, false);
            collector.collectData(tickCounter, attacking);

            List<Integer> ticks = attackTicks.get(playerName);
            if (ticks != null) {
                Iterator<Integer> iterator = ticks.iterator();
                while (iterator.hasNext()) {
                    int attackTick = iterator.next();
                    if (tickCounter - attackTick >= 60) {
                        iterator.remove();
                    }
                }
            }
            // 重置攻击状态
            isAttacking.put(playerName, false);
        }
    }

    @EventHandler
    public void onEntityDamage(EntityDamageByEntityEvent event) {
        Entity damager = event.getDamager();
        Entity damagee = event.getEntity();

        if (damager instanceof Player && damagee instanceof Player) {
            Player attacker = (Player) damager;
            List<Integer> ticks = attackTicks.get(attacker.getName());
            if (ticks != null) {
                ticks.add(tickCounter);
                isAttacking.put(attacker.getName(), true);
            }
        }
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        String playerName = event.getPlayer().getName();
        DataCollector collector = collectors.remove(playerName);
        attackTicks.remove(playerName);
        isAttacking.remove(playerName);
        if (collector != null) {
            collector.saveData();
        }
    }

    private class DataCollector {
        private final String playerName;
        private final long startTime;
        private BufferedWriter writer;
        private final File dataFile;
        private Vector lastPosition;
        private long lastTime;

        public DataCollector(String playerName) {
            this.playerName = playerName;
            this.startTime = System.currentTimeMillis();
            this.lastPosition = null;
            this.lastTime = startTime;

            this.dataFile = new File(getDataFolder(), "data-" + startTime + ".csv");
            try {
                writer = new BufferedWriter(new FileWriter(dataFile));
                writer.write("Tick,Time,PlayerPitch,PlayerYaw,NearestPlayerPitch,NearestPlayerYaw,SpeedToXAxis,SpeedToHorizontal,SpeedMagnitude,IsAttacking\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void collectData(int tickCounter, boolean isAttacking) {
            Player player = Bukkit.getPlayer(playerName);
            if (player == null) return;

            long currentTime = System.currentTimeMillis();
            Vector currentPosition = player.getLocation().toVector();
            double playerPitch = player.getLocation().getPitch();
            double playerYaw = player.getLocation().getYaw();

            Vector viewDirection = new Vector(
                    -Math.sin(Math.toRadians(playerYaw)) * Math.cos(Math.toRadians(playerPitch)),
                    -Math.sin(Math.toRadians(playerPitch)),
                    Math.cos(Math.toRadians(playerYaw)) * Math.cos(Math.toRadians(playerPitch))
            );

            double playerDirectionHorizontal = Math.sqrt(Math.pow(viewDirection.getX(), 2) + Math.pow(viewDirection.getZ(), 2));
            double playerDirectionYaw = Math.toDegrees(Math.atan2(viewDirection.getZ(), viewDirection.getX()));
            double playerDirectionPitch = Math.toDegrees(Math.atan2(viewDirection.getY(), playerDirectionHorizontal));

            Player nearestPlayer = findNearestPlayer(player);
            double nearestPlayerPitch = 0;
            double nearestPlayerYaw = 0;

            if (nearestPlayer != null) {
                Vector directionToNearest = nearestPlayer.getLocation().toVector().subtract(currentPosition);
                double horizontalDistanceToNearest = Math.sqrt(directionToNearest.getX() * directionToNearest.getX() + directionToNearest.getZ() * directionToNearest.getZ());
                nearestPlayerPitch = Math.toDegrees(Math.atan2(directionToNearest.getY(), horizontalDistanceToNearest));
                nearestPlayerYaw = Math.toDegrees(Math.atan2(directionToNearest.getZ(), directionToNearest.getX()));
            }

            double speedMagnitude = 0;
            double speedToXAxis = 0;
            double speedToHorizontal = 0;

            if (lastPosition != null) {
                long timeDelta = currentTime - lastTime;
                Vector displacement = currentPosition.clone().subtract(lastPosition);
                speedMagnitude = displacement.length() / (timeDelta / 1000.0);

                speedToXAxis = Math.toDegrees(Math.atan2(displacement.getZ(), displacement.getX()));
                speedToHorizontal = Math.toDegrees(Math.atan2(displacement.getY(), Math.sqrt(displacement.getX() * displacement.getX() + displacement.getZ() * displacement.getZ())));
            }

            lastPosition = currentPosition;
            lastTime = currentTime;

            try {
                writer.write(String.format("%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%b\n",
                        tickCounter, currentTime - startTime, playerDirectionPitch, playerDirectionYaw, nearestPlayerPitch, nearestPlayerYaw,
                        speedToXAxis, speedToHorizontal, speedMagnitude, isAttacking));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void saveData() {
            try {
                if (writer != null) {
                    writer.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private Player findNearestPlayer(Player player) {
            return Bukkit.getOnlinePlayers().stream()
                    .filter(p -> !p.equals(player))
                    .min(Comparator.comparingDouble(p -> player.getLocation().distance(p.getLocation())))
                    .orElse(null);
        }
    }
}
