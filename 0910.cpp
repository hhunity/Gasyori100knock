
スループット、レイテンシーの数字出せるように
テキストエディタ選択
⭐️テストまわす
⭐️バッファリングが、溜まった時と、たまらなかった時で
きちんと画像見れているか確認
⭐️同一画像比較の場合にメモリーどーなってるか確認
⭐️100でエラーになった理由


⭐️detectの確認
-フィルター周り確認。
-ctx周りで競合しないか確認。
⭐️corretの確認
-2値化変更できるように。
-inとoutを別メモリに







cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& range){
    for (int y = range.start; y < range.end; y++) {
        const uchar* pSrc = src.ptr<uchar>(y);
        uchar* pDst = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; x++)
            pDst[x] = (pSrc[x] > thresh ? maxval : 0);
    }
});



using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

public sealed class UiBatchScope : IDisposable
{
    private readonly Control _target;
    private bool _resumed;

    public UiBatchScope(Control target)
    {
        _target = target ?? throw new ArgumentNullException(nameof(target));
        _target.SuspendLayout();                 // レイアウト停止
        SendMessage(_target.Handle, WM_SETREDRAW, IntPtr.Zero, IntPtr.Zero); // 描画停止
    }

    public void Dispose()
    {
        if (_resumed) return;
        SendMessage(_target.Handle, WM_SETREDRAW, (IntPtr)1, IntPtr.Zero);   // 描画再開
        _target.ResumeLayout(performLayout: true);                           // レイアウト再開
        _target.Invalidate(true);                                            // 全体を無効化
        _target.Update();                                                    // ここで“一気に描画”
        _resumed = true;
    }

    private const int WM_SETREDRAW = 0x000B;
    [DllImport("user32.dll")] private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
}
using (new UiBatchScope(panel1))
{
    // ここでテキストボックスやボタンの状態を“全部まとめて”最終状態に変更する
    textBox1.Text = "準備完了";
    buttonRun.Enabled = false;
    buttonStop.Enabled = true;
    // …ほかの Enabled/Text/Visible 変更を全部ここで
} // スコープを抜けた瞬間に、一度だけ描画されて“ドンッ”と切り替わる

// バックグラウンドスレッドから
this.BeginInvoke((Action)(() => {
    // UI が空いたタイミングで実行
    textBox1.AppendText("非同期呼び出し\n");
}));



richTextBox1.ReadOnly = true;          // 入力不可にする
richTextBox1.WordWrap = false;         // 折り返しオフで軽く
richTextBox1.DetectUrls = false;       // URL自動リンク切る
richTextBox1.ShortcutsEnabled = false; // Ctrl+Zなど不要なら切る
richTextBox1.HideSelection = true;     // 選択反転を見えなくする

void AppendLog(string msg, Color color)
{
    if (richTextBox1.IsDisposed) return;

    richTextBox1.BeginInvoke((Action)(() =>
    {
        // カーソルを末尾に移動
        richTextBox1.SelectionStart = richTextBox1.TextLength;
        richTextBox1.SelectionLength = 0;

        // 書式を設定
        richTextBox1.SelectionColor = color;

        // 追加
        richTextBox1.AppendText(msg + Environment.NewLine);

        // 自動スクロール
        richTextBox1.ScrollToCaret();

        // 長さ制限（例: 20万文字でカット）
        if (richTextBox1.TextLength > 200_000)
        {
            richTextBox1.Text = richTextBox1.Text[^100_000..];
        }
    }));
}

bool IsAtBottom(RichTextBox rtb)
{
    int visibleLines = rtb.Height / TextRenderer.MeasureText("A", rtb.Font).Height;
    int firstVisible = rtb.GetLineFromCharIndex(rtb.GetCharIndexFromPosition(new Point(1, 1)));
    int lastVisible = rtb.GetLineFromCharIndex(rtb.GetCharIndexFromPosition(new Point(1, rtb.Height - 1)));
    int totalLines = rtb.GetLineFromCharIndex(rtb.TextLength) + 1;
    return lastVisible >= totalLines - 1;
}

void AppendLog(string msg)
{
    bool autoscroll = IsAtBottom(richTextBox1);

    richTextBox1.AppendText(msg + Environment.NewLine);

    if (autoscroll)
    {
        richTextBox1.SelectionStart = richTextBox1.TextLength;
        richTextBox1.ScrollToCaret();
    }
}















Invalidate() …「あとで描いて」。キューに入る（合成され、まとめて1回になる）
	•	Update() …「今すぐ未処理の WM_PAINT を処理」
	•	Refresh() … Invalidate() → Update() を連続で呼ぶ（＝即時再描画）


this.SetStyle(ControlStyles.AllPaintingInWmPaint |
              ControlStyles.UserPaint |
              ControlStyles.OptimizedDoubleBuffer, true);
this.UpdateStyles();

public partial class CanvasPanel : Panel
{
    private readonly Timer _frameTimer = new Timer(); // WinForms タイマー（UIスレッド）

    public CanvasPanel()
    {
        SetStyle(ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.UserPaint |
                 ControlStyles.OptimizedDoubleBuffer, true);
        UpdateStyles();

        _frameTimer.Interval = 16; // 約60FPS
        _frameTimer.Tick += (_, __) => Invalidate(); // 毎フレーム1回だけ無効化
        _frameTimer.Start();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        // ここで“そのフレームで必要なものを全部描く”
        // e.Graphics.DrawImage(...); e.Graphics.DrawLines(...); など
        base.OnPaint(e);
    }
}







#pragma pack(push, 1)   // 1バイト境界に詰める
struct A {
    char c;     // 1 byte
    int  i;     // 4 byte
};
#pragma pack(pop)

[StructLayout(LayoutKind.Sequential, Pack = 1)]
struct A {
    public byte c;
    public int i;
}

