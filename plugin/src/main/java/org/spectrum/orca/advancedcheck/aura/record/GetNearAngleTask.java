package org.spectrum.orca.advancedcheck.aura.record;

import org.bukkit.entity.Player;
import org.bukkit.util.Vector;
import org.bukkit.Location;

import java.util.List;

public class GetNearAngleTask {

    // 获取玩家视角与目标玩家之间的夹角
    // 获取玩家当前视线和目标玩家之间的夹角
    public float getAngle(Player player) {
        if (!player.isOnline()){return 0;}//防止玩家下线

        Player target = findNearPlayer(player);

        if (target == null){return 0;}//附近没人
        // 获取玩家当前视线的方向
        Vector playerLineOfSight = player.getLocation().getDirection();

        // 获取目标玩家的中心位置
        Location middle = getPlayerCenter(target);

        Vector targetLocation = middle.toVector();

        // 计算从玩家到目标玩家的方向向量
        Vector directionToTarget = targetLocation.subtract(player.getEyeLocation().toVector()).normalize();

        // 计算两个向量之间的夹角
        double dotProduct = playerLineOfSight.dot(directionToTarget);
        dotProduct = Math.min(1.0, Math.max(-1.0, dotProduct));
        double angleInRadians = Math.acos(dotProduct);

        // 将弧度转化为度数
        //double angleInDegrees = Math.toDegrees(angleInRadians);

        //因为度一般比较大，使用弧度制
        angleInRadians = Math.abs(angleInRadians);
        return (float)angleInRadians;
    }

    // 获取附近的目标玩家
    private Player findNearPlayer(Player player) {
        // 获取玩家的位置
        Vector playerLocation = player.getLocation().toVector();

        // 获取玩家周围的所有玩家
        List<Player> nearbyPlayers = player.getWorld().getPlayers();

        Player nearestPlayer = null;
        double minDistance = Double.MAX_VALUE;

        for (Player nearbyPlayer : nearbyPlayers) {
            // 排除自己
            if (nearbyPlayer.equals(player)) {
                continue;
            }

            // 计算距离
            double distance = playerLocation.distance(nearbyPlayer.getLocation().toVector());

            // 如果在10格内且比当前最近玩家更近
            if (distance <= 10 && distance < minDistance) {
                nearestPlayer = nearbyPlayer;
                minDistance = distance;
            }
        }

        return nearestPlayer;
    }
    // 获取玩家中心坐标的替代方案
    public Location getPlayerCenter(Player player) {
        // 1.8.9 中玩家基础高度为 1.8（潜行时 1.65）
        double height = player.isSneaking() ? 1.65 : 1.8;
        return player.getLocation().add(0, height / 2, 0);
    }

}
