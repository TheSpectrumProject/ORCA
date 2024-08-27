package ac.grim.grimac;

import org.bukkit.Bukkit;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class PunishCommandExecutor implements CommandExecutor {

    private final JavaPlugin plugin;

    public PunishCommandExecutor(JavaPlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length < 3) {
            sender.sendMessage("Usage: /orca-punish [Ban/Kick/TempBan] [PlayerName] [REASON] [Duration (optional for TempBan)]");
            return false;
        }

        String action = args[0];
        String playerName = args[1];
        String reason = args[2];
        String duration = (args.length >= 4) ? args[3] : "1d"; // Default duration is 1 day for TempBan

        Player target = Bukkit.getPlayer(playerName);

        if (target == null) {
            sender.sendMessage("Player " + playerName + " not found.");
            return false;
        }

        // 执行惩罚操作
        switch (action.toLowerCase()) {
            case "ban":
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "ban " + playerName + " ORCA Unfair advantage.");
                break;
            case "kick":
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "kick " + playerName + " ORCA Unfair advantage.");
                break;
            case "tempban":
                Bukkit.dispatchCommand(Bukkit.getConsoleSender(), "tempban " + playerName + " " + duration + " ORCA Unfair advantage.");
                break;
            default:
                sender.sendMessage("Invalid action. Use Ban/Kick/TempBan.");
                return false;
        }

        // 广播信息
        Bukkit.broadcastMessage(playerName + " was punished for cheating!");

        // 记录日志
        logPunishment(action, playerName, reason, duration);

        return true;
    }

    private void logPunishment(String action, String playerName, String reason, String duration) {
        File logFile = new File(plugin.getDataFolder(), "punish-log.txt");
        String timeStamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(logFile, true))) {
            writer.write("[" + timeStamp + "] Player: " + playerName + ", Action: " + action +
                    (action.equalsIgnoreCase("tempban") ? " (Duration: " + duration + ")" : "") +
                    ", Reason: " + reason + "\n");
        } catch (IOException e) {
            plugin.getLogger().severe("Failed to write to punish-log.txt");
            e.printStackTrace();
        }
    }
}
