package ac.grim.grimac;
//START
import org.bukkit.Bukkit;
//END
import org.bukkit.plugin.java.JavaPlugin;



public final class GrimAC extends JavaPlugin {

    //START
    private final ORCADLDataHandler dataHandler = new ORCADLDataHandler();
    //END
    @Override
    public void onLoad() {
        GrimAPI.INSTANCE.load(this);
    }

    @Override
    public void onDisable() {
        GrimAPI.INSTANCE.stop(this);
    }

    @Override
    public void onEnable() {
        GrimAPI.INSTANCE.start(this);
        //START
        // 确保数据文件夹存在
        if (!getDataFolder().exists()) {
            getDataFolder().mkdirs();
        }

        // 注册命令
        this.getCommand("orca-dl").setExecutor(dataHandler);

        // 注册事件监听器
        Bukkit.getPluginManager().registerEvents(dataHandler, this);

        // 启动定时任务
        Bukkit.getScheduler().runTaskTimer(this, dataHandler::onTick, 0, 1);
        //Punish Command

        this.getCommand("orca-punish").setExecutor(new PunishCommandExecutor(this));

        //END
    }












    }