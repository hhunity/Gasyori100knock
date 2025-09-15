// 
OK、**「4枚平均→厳密FIFO→一定周期で送信」**にまとめた送信ステージの実装例です（.NET Framework 4.8 / System.Threading.Channels / System.Buffers）。
	•	取得→処理までは今まで通り（TxJob{ Seq, Payload, Length } を _txCh に流す）
	•	送信はタイマー駆動で、**次に送るべきグループ（4枚）**が揃っていれば平均して送信
	•	揃っていない Tick の扱いはポリシーで選べます：Skip / RepeatLast / Wait

ポイント
	•	厳密順序を守るため、取得側の captureCh は FullMode = Wait を推奨（欠番が出ると以降が詰みます）。
	•	RepeatLast は外部が等間隔受信を期待する場合に有効（等間隔は保つ／情報は最新ではない可能性）。
欠落が許容されるなら Skip、**厳密に“新しいデータのみ”**を送りたいなら Wait/Skip を選択。
	•	Timer は Windows の都合で数十µs程度のジッタがあります。さらに詰めるなら、Tick内で
	•	Stopwatch ベースで短スリープ＋短スピン（高精度待ち）
	•	あるいはハード側の等時性転送（Isochronous）を使う
なども検討してください。

必要なら、この送信ステージをあなたの既存パイプライン全体ファイルにマージした完全版を出します（eBUS/USB 実装の差し替え位置込み）。



Pipeline クラスの一部（送信ステージ：4枚平均 + 厳密順序 + 周期送信）

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

public enum OnMissPolicy
{
    Skip,        // 揃ってなければ送らない（周期は維持、欠落あり）
    RepeatLast,  // 前回の平均結果を再送（等間隔を維持）
    Wait         // 揃うまで送らない（=Skipと同義だが将来拡張用）
}

public sealed class Pipeline
{
    // ……（既存フィールドは省略）……

    private const int AggregateGroupSize = 4;
    private readonly Dictionary<int, List<TxJob>> _groups = new Dictionary<int, List<TxJob>>(); // groupId -> 受信済4枚
    private int _nextGroupToSend = 0;           // 次に送るべき groupId（0,1,2,...）
    private readonly object _aggLock = new object(); // _groups / _nextGroupToSend の排他用

    private System.Threading.Timer _tick;       // 周期送信用タイマー
    private volatile bool _tickSending;         // 再入防止
    private OnMissPolicy _missPolicy = OnMissPolicy.RepeatLast;
    private byte[] _lastAvg;                    // RepeatLast 用の「前回平均」保持（1枚ぶん）

    private Task _txTask;                       // txCh を読み取って _groups を埋めるタスク
    private readonly CancellationTokenSource _cts = new CancellationTokenSource();
    private readonly IUsbTransport _usb;
    private readonly Channel<TxJob> _txCh;

    // ★ 呼び出し口：周期送信の開始
    public void StartUsbTxLoop_Aggregate4_Periodic(TimeSpan period, OnMissPolicy missPolicy = OnMissPolicy.RepeatLast)
    {
        _missPolicy = missPolicy;

        // 1) txCh -> _groups への集約（揃うかどうかに関わらず溜めるだけ）
        _txTask = Task.Run(async () =>
        {
            try
            {
                while (await _txCh.Reader.WaitToReadAsync(_cts.Token))
                {
                    TxJob job;
                    while (_txCh.Reader.TryRead(out job))
                    {
                        int gid = (job.Seq - 1) / AggregateGroupSize;
                        lock (_aggLock)
                        {
                            if (!_groups.TryGetValue(gid, out var list))
                            {
                                list = new List<TxJob>(AggregateGroupSize);
                                _groups[gid] = list;
                            }
                            list.Add(job);
                        }
                    }
                }
            }
            catch (OperationCanceledException) { }
        }, _cts.Token);

        // 2) 周期タイマー起動（Tick毎に1グループだけ処理）
        _tick = new System.Threading.Timer(_ =>
        {
            if (_tickSending) return;      // 再入防止（ロングI/O中に次Tickが来るのを防ぐ）
            _tickSending = true;
            _ = Task.Run(async () =>
            {
                try { await OnTickAsync(); }
                finally { _tickSending = false; }
            });
        }, null, dueTime: period, period: period);
    }

    // ★ Tick処理：次の番のグループが揃っていれば平均して送信。揃っていなければポリシーに従う
    private async Task OnTickAsync()
    {
        List<TxJob> list = null;
        int groupId = -1;

        // 1) 次の番のグループを引き当て（重い処理はロック外で実施）
        lock (_aggLock)
        {
            if (_groups.TryGetValue(_nextGroupToSend, out var cand) && cand.Count == AggregateGroupSize)
            {
                list = cand;
                groupId = _nextGroupToSend;
                _groups.Remove(_nextGroupToSend);   // 取り外してロックを短時間化
                _nextGroupToSend++;
            }
        }

        var pool = ArrayPool<byte>.Shared;

        if (list != null)
        {
            // 2) 平均を作って送る
            int len = list[0].Length;          // 同一長前提（異なる場合は最小長などに合わせる処理を追加）
            byte[] avg = pool.Rent(len);
            try
            {
                // 画素ごとに 4 枚平均（Mono8想定、四捨五入）
                for (int i = 0; i < len; i++)
                {
                    int sum = list[0].Payload[i] + list[1].Payload[i] + list[2].Payload[i] + list[3].Payload[i];
                    avg[i] = (byte)((sum + 2) >> 2);
                }

                await _usb.SendAsync(avg, len, _cts.Token);

                // RepeatLast 用に保持（確保は一度だけ、長さが変わるなら作り直す）
                if (_missPolicy == OnMissPolicy.RepeatLast)
                {
                    if (_lastAvg == null || _lastAvg.Length != len) _lastAvg = new byte[len];
                    Buffer.BlockCopy(avg, 0, _lastAvg, 0, len);
                }
            }
            finally
            {
                // 入力4枚・平均のバッファを返却
                foreach (var j in list) pool.Return(j.Payload);
                pool.Return(avg);
            }
        }
        else
        {
            // 3) 揃っていない Tick の扱い
            switch (_missPolicy)
            {
                case OnMissPolicy.Skip:
                case OnMissPolicy.Wait:
                    // 何もしない（等間隔は維持、送信は欠落）
                    break;

                case OnMissPolicy.RepeatLast:
                    // 前回結果を再送（一定周期を維持しつつ見かけの欠落を回避）
                    var buf = _lastAvg;
                    if (buf != null && buf.Length > 0)
                    {
                        await _usb.SendAsync(buf, buf.Length, _cts.Token);
                    }
                    break;
            }
        }
    }

    // ★ 優雅停止（すべて流し切ってから終了）
    public async Task StopGracefullyAsync()
    {
        // （取得停止→処理完了→txCh close）は既存ロジックに合わせて呼ぶ想定
        _tick?.Change(Timeout.Infinite, Timeout.Infinite);
        _tick?.Dispose();

        _txCh.Writer.TryComplete();               // 送信前段への新規投入を止める
        if (_txTask != null) await _txTask;       // 集約タスク終了待ち

        // 残グループの未返却バッファを解放（厳密順序では送らない）
        var pool = ArrayPool<byte>.Shared;
        lock (_aggLock)
        {
            foreach (var kv in _groups)
                foreach (var j in kv.Value) pool.Return(j.Payload);
            _groups.Clear();
        }
    }

    public void Stop() => _cts.Cancel();
}