package org.spectrum.orca;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerInteractEntityEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ORCA extends JavaPlugin implements CommandExecutor, Listener {

    private Map<String, DataCollector> collectors = new HashMap<>();

    @Override
    public void onEnable() {
        this.getCommand("orca").setExecutor(this);
        Bukkit.getPluginManager().registerEvents(this, this);
        Bukkit.getScheduler().runTaskTimer(this, this::collectData, 0, 1);
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length < 2) {
            return false;
        }

        String subCommand = args[0];
        String playerName = args[1];
        Player player = Bukkit.getPlayer(playerName);

        if (player == null) {
            sender.sendMessage("Player not found.");
            return true;
        }

        switch (subCommand) {
            case "collectdata":
                if (args.length == 3 && args[2].equals("start")) {
                    startCollectingData(playerName);
                    sender.sendMessage("Started collecting data for " + playerName);
                } else if (args.length == 3 && args[2].equals("stop")) {
                    stopCollectingData(playerName);
                    sender.sendMessage("Stopped collecting data for " + playerName);
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
    }

    private void stopCollectingData(String playerName) {
        DataCollector collector = collectors.remove(playerName);
        if (collector != null) {
            collector.saveData();
        }
    }

    private void collectData() {
        for (DataCollector collector : collectors.values()) {
            collector.collectData();
        }
    }

    @EventHandler
    public void onPlayerInteract(PlayerInteractEvent event) {
        Player player = event.getPlayer();
        DataCollector collector = collectors.get(player.getName());
        if (collector != null) {
            collector.updateClickState(event);
        }
    }

    @EventHandler
    public void onPlayerInteractEntity(PlayerInteractEntityEvent event) {
        Player player = event.getPlayer();
        DataCollector collector = collectors.get(player.getName());
        if (collector != null) {
            collector.updateClickState(event);
        }
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        String playerName = event.getPlayer().getName();
        DataCollector collector = collectors.remove(playerName);
        if (collector != null) {
            collector.saveData();
        }
    }

    private class DataCollector {
        private String playerName;
        private long startTime;
        private BufferedWriter writer;
        private int clickState;

        public DataCollector(String playerName) {
            this.playerName = playerName;
            this.startTime = System.currentTimeMillis();
            try {
                File file = new File(getDataFolder(), "data-" + startTime + ".csv");
                writer = new BufferedWriter(new FileWriter(file));
                writer.write("Time,PlayerPitch,PlayerYaw,NearestPlayerPitch,NearestPlayerYaw," +
                        "SpeedToXAxis,SpeedToHorizontal,SpeedMagnitude,ClickState\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public void collectData() {
            Player player = Bukkit.getPlayer(playerName);
            if (player == null) return;

            double playerPitch = player.getLocation().getPitch();
            double playerYaw = player.getLocation().getYaw();
            double nearestPlayerPitch = 0;
            double nearestPlayerYaw = 0;

            Player nearestPlayer = findNearestPlayer(player);
            if (nearestPlayer != null) {
                double dx = nearestPlayer.getLocation().getX() - player.getLocation().getX();
                double dz = nearestPlayer.getLocation().getZ() - player.getLocation().getZ();
                double angleToNearestPlayer = Math.atan2(dz, dx) * (180 / Math.PI);

                nearestPlayerPitch = nearestPlayer.getLocation().getPitch();
                nearestPlayerYaw = angleToNearestPlayer;
            }

            double speedX = player.getVelocity().getX();
            double speedY = player.getVelocity().getY();
            double speedZ = player.getVelocity().getZ();
            double speedMagnitude = player.getVelocity().length();

            double speedToXAxis = Math.toDegrees(Math.atan2(speedZ, speedX));
            double speedToHorizontal = Math.toDegrees(Math.atan2(speedY, Math.sqrt(speedX * speedX + speedZ * speedZ)));

            try {
                writer.write(String.format("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n",
                        System.currentTimeMillis() - startTime, playerPitch, playerYaw, nearestPlayerPitch, nearestPlayerYaw,
                        speedToXAxis, speedToHorizontal, speedMagnitude, clickState));
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

        public void updateClickState(PlayerInteractEvent event) {
            Action action = event.getAction();

            if (action == Action.LEFT_CLICK_BLOCK) {
                clickState = 3; // Left click on block
            } else if (action == Action.LEFT_CLICK_AIR) {
                clickState = 1; // Left click in the air
            } else {
                clickState = 0; // Other actions or no click
            }
        }

        public void updateClickState(PlayerInteractEntityEvent event) {
            clickState = 2; // Left click on entity
        }

        private Player findNearestPlayer(Player player) {
            Player nearestPlayer = null;
            double minDistance = Double.MAX_VALUE;
            for (Player p : Bukkit.getOnlinePlayers()) {
                if (!p.equals(player)) {
                    double distance = player.getLocation().distance(p.getLocation());
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestPlayer = p;
                    }
                }
            }
            return nearestPlayer;
        }
    }
}
