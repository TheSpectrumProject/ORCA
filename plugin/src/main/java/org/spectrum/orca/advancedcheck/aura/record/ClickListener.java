package org.spectrum.orca.advancedcheck.aura.record;

import org.bukkit.event.Listener;
import org.bukkit.event.EventHandler;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.block.Action;
import org.bukkit.entity.Player;

public class ClickListener implements Listener {

    private final RecordMain recordMain;  // 传入AuraMain以便调用recordSingle

    public ClickListener(RecordMain recordMain) {
        this.recordMain = recordMain; // 通过构造函数传入 RecordMain 实例
    }

    @EventHandler
    public void onPlayerClick(PlayerInteractEvent event) {
        Player player = event.getPlayer(); // 获取玩家
        if (event.getAction() == Action.LEFT_CLICK_AIR || event.getAction() == Action.LEFT_CLICK_BLOCK) {
            // 执行 recordSingleView() 方法
            recordMain.recordSingleView(player, "0.5"); // 通过 plugin 调用主插件类的方法
        }
    }

    @EventHandler
    public void onPlayerDamage(EntityDamageByEntityEvent event) {
        // 检查攻击者是否是玩家，并且目标是玩家
        if (event.getDamager() instanceof Player && event.getEntity() instanceof Player) {
            Player attacker = (Player) event.getDamager();
            recordMain.recordSingleView(attacker, "1"); // 通过 plugin 调用主插件类的方法
        }
    }
}
