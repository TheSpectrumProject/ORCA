package org.spectrum.orca.advancedcheck.aura.punish;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.ProtocolLibrary;
import com.comphenix.protocol.events.PacketContainer;
import com.comphenix.protocol.wrappers.WrappedChatComponent;
import org.bukkit.entity.Player;

public class ActionBarUtils {

    public static void sendActionBar(Player player, String message) {
        // 构造数据包
        PacketContainer packet = new PacketContainer(PacketType.Play.Server.CHAT);

        // 设置消息类型为 ACTION_BAR (1.8.9 中类型为 2)
        packet.getBytes().write(0, (byte) 2);

        // 将消息转换为 JSON 组件（兼容颜色代码）
        WrappedChatComponent chatComponent = WrappedChatComponent.fromJson(
                "{\"text\": \"" + message + "\"}"
        );

        // 写入数据包内容
        packet.getChatComponents().write(0, chatComponent);

        try {
            // 发送数据包给玩家
            ProtocolLibrary.getProtocolManager().sendServerPacket(player, packet);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
