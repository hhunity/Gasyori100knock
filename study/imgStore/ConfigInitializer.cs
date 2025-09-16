using System;
using System.IO;

static class ConfigInitializer
{
    private const string CompanyId = "HoshikaWorks";
    private const string ProductId = "PrinterCtrl_Pro";
    private const string UserConfigFileName = "config.json";
    private const string DefaultConfigFileName = "default.config.json";

    public static void CopyDefaultIfNeeded()
    {
        string installPath = Path.Combine(AppContext.BaseDirectory, DefaultConfigFileName);
        if (!File.Exists(installPath))
            return; // インストール先に無ければ何もしない

        string userDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            CompanyId, ProductId
        );
        Directory.CreateDirectory(userDir);

        string userPath = Path.Combine(userDir, UserConfigFileName);

        // まだ存在しなければコピー
        if (!File.Exists(userPath))
        {
            File.Copy(installPath, userPath);
        }
    }

    public static string GetUserConfigPath()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            CompanyId, ProductId, UserConfigFileName
        );
    }
}