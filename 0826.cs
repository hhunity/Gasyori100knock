•	captureCh：フレーム受け渡し（取得→処理へ）。容量小さめでバックプレッシャ。
	•	txCh：送信ジョブ（処理→USBへ）。
	•	画像処理ワーカーは ProcessorCount-1 並列。
	•	USB送信は基本1本（デバイス側が直列、スループット重視ならバッチ化）。

設計の要点（eBUS×並列×USB）
	1.	eBUS バッファはすぐ返す（Requeue）
	•	下流で非同期に触るならコピー前提。デバイス側のリングを空けないとドロップ率が上がる。
	•	メモリは MemoryPool<byte> / ArrayPool<byte> で再利用し GC 圧を削減。
	2.	Channel.CreateBounded で“詰まり対策”
	•	captureCh は 小容量（8〜32） を推奨。DropOldest で最新優先（監視・AR等で有効）。
	•	ロスが困る用途は Wait に変更し、入力側（カメラのFPS/露光）を下げるか、処理/送信を増強。
	3.	CPU 並列度
	•	ProcessorCount-1 からスタート。画像処理の内容次第で帯域/キャッシュがボトルネックになるので実測調整。
	•	各ワーカーは 再利用可能な一時バッファを持たせてもOK（ThreadStaticや ArrayPool）。
	4.	USB 送信は I/O バウンド
	•	まずは単一コンシューマで十分。帯域を使い切れない場合はバッチ送信（複数フレーム連結）や**重ね書き（パイプ）**を検討。
	•	送信結果に応じてリトライ・エラーハンドリングを実装。
	5.	優先度／落とし方
	•	「最新フレームを優先」→ captureCh を DropOldest。
	•	「結果は全件必要」→ Wait にして、入力レート制御（Exposure/FPS/スキップ）を必須に。
	•	高/低優先度があるなら チャネルを2本 用意して、ワーカーが高→低の順で読む。
	6.	ゼロコピー志向（高度）
	•	eBUSの PvBuffer を下流で直接使いたい場合、Requeue前提にできないので設計が難しくなる。
	•	“Requeueは処理完了後にする” 方式も可能だが、取得が詰まりやすい。一般にはコピーして即Requeueが安定。
	7.	計測
	•	取得→処理→送信それぞれの p50/p95/p99 レイテンシ と キュー長最大値 をログ。
	•	送信帯域（MB/s or pkt/s）とドロップ率（DropOldest時）を可視化。






using System.Buffers;
using System.Threading.Channels;
using System.Runtime.InteropServices;

record Frame(IMemoryOwner<byte> Buffer, int Width, int Height, int Stride, long Timestamp);
record TxJob(IMemoryOwner<byte> Payload, int Length, long Timestamp);

public class Pipeline
{
    readonly Channel<Frame> captureCh;
    readonly Channel<TxJob> txCh;
    readonly CancellationTokenSource cts = new();

    readonly int procDegree = Math.Max(1, Environment.ProcessorCount - 1);

    public Pipeline(int captureCapacity = 8, int txCapacity = 32)
    {
        captureCh = Channel.CreateBounded<Frame>(new BoundedChannelOptions(captureCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest, // 古いフレームを捨てて新しいのを優先（用途でWait等に変更）
            SingleWriter = true,   // 取得スレッドは1本ならtrueでロック軽減
            SingleReader = false
        });

        txCh = Channel.CreateBounded<TxJob>(new BoundedChannelOptions(txCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait, // 送信は溜めすぎない（待たせる）
            SingleWriter = false,
            SingleReader = true   // 送信は1本にして順序安定
        });
    }

    // --- 1) 取得スレッド：eBUSから受け取り、コピーして即Requeue ---
    public void StartCaptureLoop(PvStream stream /* eBUSのストリーム等 */)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                while (!cts.IsCancellationRequested)
                {
                    // eBUS: タイムアウト付きでバッファ取得
                    // var result = stream.RetrieveBuffer(out PvBuffer pvb, timeoutMs);
                    // if (result != OK) continue;

                    PvBuffer pvb = RetrieveBuffer(stream, 1000); // 擬似：あなたのAPIで置換

                    // 画像情報を取得
                    IntPtr srcPtr = pvb.GetDataPointer(); // 8/16bit等に応じて取得方法を変更
                    int width = pvb.Width;
                    int height = pvb.Height;
                    int stride = pvb.Stride; // バイト/行
                    long ts = pvb.Timestamp;

                    int bytes = stride * height;

                    // 下流で非同期処理するため**デバイスバッファは触らない**。
                    // すぐ Requeue できるよう**コピー**（ArrayPoolで最少GC）
                    var owner = MemoryPool<byte>.Shared.Rent(bytes);
                    var span = owner.Memory.Span.Slice(0, bytes);

                    unsafe
                    {
                        Buffer.MemoryCopy((void*)srcPtr, Unsafe.AsPointer(ref span[0]), bytes, bytes);
                    }

                    // eBUSに即返却（取りこぼし防止）
                    Requeue(stream, pvb);

                    // 受け渡し（満杯ならDropOldestで最新を優先/Waitに変えるのも可）
                    await captureCh.Writer.WriteAsync(new Frame(owner, width, height, stride, ts), cts.Token);
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                captureCh.Writer.TryComplete();
            }
        }, cts.Token);
    }

    // --- 2) 画像処理ワーカー：複数CPUで並列 ---
    public void StartProcessingWorkers()
    {
        for (int i = 0; i < procDegree; i++)
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    await foreach (var frame in captureCh.Reader.ReadAllAsync(cts.Token))
                    {
                        // Heavy CPU-bound image processing
                        // 入力は frame.Buffer.Memory.Span（0..(Stride*Height)）
                        var payloadOwner = ProcessFrame(frame, out int payloadLen);

                        // 上流バッファ解放（忘れない！）
                        frame.Buffer.Dispose();

                    retryWrite:
                        // 送信キューへ
                        try
                        {
                            await txCh.Writer.WriteAsync(
                                new TxJob(payloadOwner, payloadLen, frame.Timestamp), cts.Token);
                        }
                        catch
                        {
                            // 書けなければpayloadOwnerもリークしないよう解放
                            payloadOwner.Dispose();
                            throw;
                        }
                    }
                }
                catch (OperationCanceledException) { }
            }, cts.Token);
        }
    }

    // --- 3) USB送信（非同期I/O、順序維持）---
    public void StartUsbTxLoop(IUsbTransport usb)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await foreach (var job in txCh.Reader.ReadAllAsync(cts.Token))
                {
                    var span = job.Payload.Memory.Span.Slice(0, job.Length);
                    await usb.SendAsync(span, cts.Token);  // あなたのUSB実装（WinUSB/libusb等）
                    job.Payload.Dispose();                  // 送信バッファ解放
                }
            }
            catch (OperationCanceledException) { }
        }, cts.Token);
    }

    public void Stop()
    {
        cts.Cancel();
    }

    // ==== 以下はダミー/置換ポイント ====

    private PvBuffer RetrieveBuffer(PvStream s, int timeoutMs) => throw new NotImplementedException();
    private void Requeue(PvStream s, PvBuffer b) => throw new NotImplementedException();

    // 画像処理：入力Frame → 送信用payload（コピー先を返す）
    private IMemoryOwner<byte> ProcessFrame(Frame f, out int payloadLen)
    {
        // 例：そのまま転送するだけ（実際はFFT/フィルタ/特徴量など）
        int bytes = f.Stride * f.Height;
        var owner = MemoryPool<byte>.Shared.Rent(bytes);
        f.Buffer.Memory.Span.Slice(0, bytes).CopyTo(owner.Memory.Span);
        payloadLen = bytes;
        return owner;
    }
}

// 擬似インターフェース
public interface IUsbTransport
{
    Task SendAsync(ReadOnlySpan<byte> data, CancellationToken ct);
}
// 擬似eBUS型
public class PvStream { }
public class PvBuffer
{
    public int Width => 0;
    public int Height => 0;
    public int Stride => 0;
    public long Timestamp => 0;
    public IntPtr GetDataPointer() => IntPtr.Zero;
}



using System;
using System.Buffers;                 // ArrayPool<T>
using System.Runtime.InteropServices;  // Marshal.Copy
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

public sealed class Frame
{
    public byte[] Buffer { get; }
    public int Length { get; }
    public int Width { get; }
    public int Height { get; }
    public int Stride { get; }
    public long Timestamp { get; }

    public Frame(byte[] buffer, int length, int w, int h, int stride, long ts)
    {
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        Length = length;
        Width = w; Height = h; Stride = stride; Timestamp = ts;
    }
}

public sealed class TxJob
{
    public byte[] Payload { get; }
    public int Length { get; }
    public long Timestamp { get; }

    public TxJob(byte[] payload, int length, long ts)
    {
        Payload = payload ?? throw new ArgumentNullException(nameof(payload));
        Length = length; Timestamp = ts;
    }
}

public interface IUsbTransport
{
    Task SendAsync(byte[] data, int length, CancellationToken ct);
}

// ===== パイプライン本体（.NET Framework 4.8 で動く形） =====
public sealed class Pipeline : IDisposable
{
    private readonly Channel<Frame> _captureCh;
    private readonly Channel<TxJob> _txCh;
    private readonly CancellationTokenSource _cts = new CancellationTokenSource();
    private readonly IUsbTransport _usb;
    private readonly int _procDegree;

    public Pipeline(IUsbTransport usb, int captureCapacity = 8, int txCapacity = 32)
    {
        _usb = usb ?? throw new ArgumentNullException(nameof(usb));
        _procDegree = Math.Max(1, Environment.ProcessorCount - 1);

        _captureCh = Channel.CreateBounded<Frame>(
            new BoundedChannelOptions(captureCapacity)
            {
                FullMode = BoundedChannelFullMode.DropOldest, // 最新優先（用途で Wait も可）
                SingleWriter = true,   // 取得スレッド1本なら高速化
                SingleReader = false   // 処理ワーカー複数で読む
            });

        _txCh = Channel.CreateBounded<TxJob>(
            new BoundedChannelOptions(txCapacity)
            {
                FullMode = BoundedChannelFullMode.Wait, // 送信は溜めすぎず待つ
                SingleWriter = false,
                SingleReader = true   // 順序・帯域の都合で単一に
            });
    }

    // --- 1) 取得ループ（eBUS） ---
    public void StartCaptureLoop(PvStream stream)
    {
        Task.Run(async () =>
        {
            var pool = ArrayPool<byte>.Shared;

            try
            {
                while (!_cts.IsCancellationRequested)
                {
                    // eBUS 側から PvBuffer を取得（あなたのAPIで置換）
                    PvBuffer pvb = RetrieveBuffer(stream, 1000); // timeout ms

                    int width = pvb.Width;
                    int height = pvb.Height;
                    int stride = pvb.Stride;
                    long ts = pvb.Timestamp;
                    int bytes = stride * height;
                    IntPtr src = pvb.GetDataPointer();

                    // すぐ Requeue するため下流用にコピー確保（ArrayPoolでGC圧を下げる）
                    byte[] dst = pool.Rent(bytes);
                    try
                    {
                        Marshal.Copy(src, dst, 0, bytes); // IntPtr → byte[] コピー
                    }
                    catch
                    {
                        pool.Return(dst);
                        Requeue(stream, pvb);
                        throw;
                    }

                    Requeue(stream, pvb); // eBUS バッファは即返却

                    // captureCh が満杯なら空くまで“だけ”待つ（処理完了は待たない）
                    await _captureCh.Writer.WriteAsync(
                        new Frame(dst, bytes, width, height, stride, ts), _cts.Token);
                }
            }
            catch (OperationCanceledException) { /* stop */ }
            finally
            {
                _captureCh.Writer.TryComplete();
            }
        }, _cts.Token);
    }

    // --- 2) 画像処理ワーカー（並列） ---
    public void StartProcessingWorkers()
    {
        for (int i = 0; i < _procDegree; i++)
        {
            Task.Run(async () =>
            {
                var pool = ArrayPool<byte>.Shared;

                try
                {
                    // net48 では await foreach 使えない → WaitToReadAsync + TryRead で回す
                    while (await _captureCh.Reader.WaitToReadAsync(_cts.Token))
                    {
                        Frame f;
                        while (_captureCh.Reader.TryRead(out f))
                        {
                            // Heavy CPU 処理 → 送信用ペイロード生成（例ではそのままコピー）
                            // 実際はここでフィルタ/FFT/特徴抽出などを行い、必要サイズのバッファに書く
                            byte[] payload = pool.Rent(f.Length);
                            Buffer.BlockCopy(f.Buffer, 0, payload, 0, f.Length);

                            // capture 側のバッファを早めに返却
                            pool.Return(f.Buffer);

                            // 送信キューが満杯なら空くまで待つ
                            await _txCh.Writer.WriteAsync(
                                new TxJob(payload, f.Length, f.Timestamp), _cts.Token);
                        }
                    }
                }
                catch (OperationCanceledException) { /* stop */ }
            }, _cts.Token);
        }
    }

    // --- 3) USB 送信ループ ---
    public void StartUsbTxLoop()
    {
        Task.Run(async () =>
        {
            var pool = ArrayPool<byte>.Shared;

            try
            {
                while (await _txCh.Reader.WaitToReadAsync(_cts.Token))
                {
                    TxJob job;
                    while (_txCh.Reader.TryRead(out job))
                    {
                        try
                        {
                            await _usb.SendAsync(job.Payload, job.Length, _cts.Token);
                        }
                        finally
                        {
                            pool.Return(job.Payload);
                        }
                    }
                }
            }
            catch (OperationCanceledException) { /* stop */ }
        }, _cts.Token);
    }

    public void Stop() => _cts.Cancel();

    public void Dispose()
    {
        _cts.Cancel();
        _cts.Dispose();
    }

    // ==== あなたの eBUS API に置き換えてください ====
    private PvBuffer RetrieveBuffer(PvStream s, int timeoutMs) { throw new NotImplementedException(); }
    private void Requeue(PvStream s, PvBuffer b) { throw new NotImplementedException(); }
}

// ==== 擬似 eBUS 型（置換対象） ====
public sealed class PvStream { }
public sealed class PvBuffer
{
    public int Width { get; private set; }
    public int Height { get; private set; }
    public int Stride { get; private set; }
    public long Timestamp { get; private set; }
    public IntPtr GetDataPointer() { return IntPtr.Zero; }
}
