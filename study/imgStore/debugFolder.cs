using System;
using System.IO;

static class DebugFolders
{
    // ユーザーに見せるアプリ名は別管理（About画面やショートカット用）
    // フォルダは改名の影響を受けない固定IDにする
    private const string CompanyId = "HoshikaWorks";
    private const string ProductId = "PrinterCtrl_Pro"; // ← 将来アプリ名変えても変えない

    public static string GetUserDebugDir()
    {
        string root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string dir  = Path.Combine(root, CompanyId, ProductId, "Debug");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static string GetSharedDebugDir()
    {
        // 全ユーザー共通（要: 権限に注意。書き込みは通常ユーザーでOK）
        string root = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData); // C:\ProgramData
        string dir  = Path.Combine(root, CompanyId, ProductId, "Debug");
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static string GetTempDir()
    {
        string root = Path.GetTempPath(); // C:\Users\<User>\AppData\Local\Temp\
        string dir  = Path.Combine(root, CompanyId, ProductId);
        Directory.CreateDirectory(dir);
        return dir;
    }
}