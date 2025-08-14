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

// Program.cs  (.NET Framework 4.8)
// 事前に NuGet: System.Threading.Channels, System.Buffers を追加

using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace Net48PipelineSample
{
    // ========= ユーティリティ DTO =========
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
            Length = length; Width = w; Height = h; Stride = stride; Timestamp = ts;
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

    // ========= 初回処理で作る共有状態 =========
    public sealed class InitState
    {
        public readonly byte[] Lut;   // 例: LUT（Mono8想定のダミー）
        public readonly int ParamA;   // 例: 係数
        public readonly double[] Kernel; // 例: 簡易カーネル
        public InitState(byte[] lut, int a, double[] kernel)
        {
            Lut = lut; ParamA = a; Kernel = kernel;
        }
    }

    // ========= USB 抽象 =========
    public interface IUsbTransport
    {
        Task SendAsync(byte[] data, int length, CancellationToken ct);
    }

    // ダミー実装（コンソールに出力するだけ）
    public sealed class DummyUsbTransport : IUsbTransport
    {
        public Task SendAsync(byte[] data, int length, CancellationToken ct)
        {
            // 実機では WinUSB / libusb / 仮想COM 等へ置換
            Console.WriteLine($"[USB] {length} bytes sent (ts=n/a)");
            return Task.CompletedTask;
        }
    }

    // ========= eBUS 側のダミー =========
    public sealed class PvStream { }
    public sealed class PvBuffer
    {
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Stride { get; private set; }
        public long Timestamp { get; private set; }
        private IntPtr _ptr;

        public PvBuffer(int w, int h)
        {
            Width = w; Height = h; Stride = w; Timestamp = DateTime.UtcNow.Ticks;
            // デモ用にGCピン留めした配列を IntPtr に見せかけても良いが、ここでは簡略化
            _ptr = IntPtr.Zero;
        }
        public IntPtr GetDataPointer() => _ptr;
    }

    // ========= パイプライン本体 =========
    public sealed class Pipeline : IDisposable
    {
        private readonly Channel<Frame> _captureCh;
        private readonly Channel<TxJob> _txCh;
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private readonly IUsbTransport _usb;
        private readonly int _procDegree;

        // 初回処理ゲートと共有状態
        private readonly TaskCompletionSource<bool> _initGate = new TaskCompletionSource<bool>();
        private int _initStarted = 0;      // 0=未開始,1=開始済み
        private InitState _state;          // 初回処理で設定 → 以後は読み取り専用

        public Pipeline(IUsbTransport usb, int captureCapacity = 8, int txCapacity = 32, int? degree = null,
                        BoundedChannelFullMode capFullMode = BoundedChannelFullMode.DropOldest)
        {
            _usb = usb ?? throw new ArgumentNullException(nameof(usb));
            _procDegree = degree ?? Math.Max(1, Environment.ProcessorCount - 1);

            _captureCh = Channel.CreateBounded<Frame>(
                new BoundedChannelOptions(captureCapacity)
                {
                    FullMode = capFullMode,    // 最新優先なら DropOldest / 欠落NGなら Wait
                    SingleWriter = true,       // 取得スレッドが1本なら true
                    SingleReader = false       // 処理ワーカーが複数読むので false
                });

            _txCh = Channel.CreateBounded<TxJob>(
                new BoundedChannelOptions(txCapacity)
                {
                    FullMode = BoundedChannelFullMode.Wait, // 送信は溜めすぎない
                    SingleWriter = false,
                    SingleReader = true
                });
        }

        // ---- 1) eBUS 取得ループ（実機APIに差し替えてください）----
        public void StartCaptureLoop(PvStream stream, int width, int height, int mockFps = 60)
        {
            Task.Run(async () =>
            {
                var pool = ArrayPool<byte>.Shared;
                var periodMs = Math.Max(1, 1000 / Math.Max(1, mockFps)); // デモ用FPS

                try
                {
                    while (!_cts.IsCancellationRequested)
                    {
                        // 実機：RetrieveBuffer(out PvBuffer pvb, timeoutMs) → pvb.GetDataPointer()
                        // ここではデモとして、ダミー画像を生成して詰める
                        int stride = width;
                        int bytes = stride * height;

                        // ダミー: 画像データをプール配列に書く
                        byte[] dst = pool.Rent(bytes);
                        for (int i = 0; i < bytes; i++) dst[i] = (byte)(i & 0xFF);
                        long ts = DateTime.UtcNow.Ticks;

                        // captureCh が満杯なら空くまで“だけ”待つ（処理完了は待たない）
                        await _captureCh.Writer.WriteAsync(new Frame(dst, bytes, width, height, stride, ts), _cts.Token);

                        await Task.Delay(periodMs, _cts.Token); // デモ用：フレーム間隔
                    }
                }
                catch (OperationCanceledException) { }
                finally
                {
                    _captureCh.Writer.TryComplete();
                }
            }, _cts.Token);
        }

        // ---- 2) 画像処理ワーカー（初回処理 → 以降は共有状態を使用）----
        public void StartProcessingWorkers()
        {
            for (int i = 0; i < _procDegree; i++)
            {
                Task.Run(async () =>
                {
                    var pool = ArrayPool<byte>.Shared;

                    try
                    {
                        while (await _captureCh.Reader.WaitToReadAsync(_cts.Token))
                        {
                            Frame f;
                            while (_captureCh.Reader.TryRead(out f))
                            {
                                // 初回担当判定（1回だけ成立）
                                if (Interlocked.CompareExchange(ref _initStarted, 1, 0) == 0)
                                {
                                    try
                                    {
                                        // --- 初回専用処理：共有状態を構築 ---
                                        _state = await DoFirstFrameInitializationAsync(f, _cts.Token);
                                        // 共有状態の公開完了
                                        _initGate.TrySetResult(true);
                                    }
                                    catch (Exception ex)
                                    {
                                        _initGate.TrySetException(ex);
                                        throw;
                                    }

                                    // 初回フレームも通常処理に流す（不要なら捨てても可）
                                    await ProcessNormallyUsingStateAsync(f, _cts.Token);
                                }
                                else
                                {
                                    // 初期化完了まで待つ（可視性保証）
                                    await _initGate.Task;
                                    await ProcessNormallyUsingStateAsync(f, _cts.Token);
                                }
                            }
                        }
                    }
                    catch (OperationCanceledException) { }
                }, _cts.Token);
            }
        }

        // ---- 3) USB 送信ループ ----
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
                catch (OperationCanceledException) { }
            }, _cts.Token);
        }

        // ---- 停止/破棄 ----
        public void Stop() => _cts.Cancel();
        public void Dispose()
        {
            _cts.Cancel();
            _cts.Dispose();
        }

        // ========== 初回処理/通常処理の中身（ダミーを用意。実装を差し替えてOK） ==========

        // 初回処理：フレームを使って LUT/係数/カーネルなどを作る
        private Task<InitState> DoFirstFrameInitializationAsync(Frame f, CancellationToken ct)
        {
            // 例：Mono8 LUTを恒等に、パラメータ/カーネルもダミーで作成
            var lut = new byte[256];
            for (int i = 0; i < lut.Length; i++) lut[i] = (byte)i;

            int paramA = 42;
            var kernel = new double[] { 0, 1, 0, 1, -4, 1, 0, 1, 0 }; // 3x3 Laplacian風（ダミー）

            // 実際は f.Buffer を解析して補正係数を推定する等
            return Task.FromResult(new InitState(lut, paramA, kernel));
        }

        // 以降の通常処理：共有状態を使って処理→送信キューへ
        private async Task ProcessNormallyUsingStateAsync(Frame f, CancellationToken ct)
        {
            var s = _state;                      // ローカルへ掴む
            var pool = ArrayPool<byte>.Shared;

            byte[] payload = pool.Rent(f.Length);
            try
            {
                // 例1: LUT適用（Mono8前提のダミー）
                ApplyLutMono8(f.Buffer, payload, f.Length, s.Lut);

                // 例2: ダミー畳み込み（実際は幅/高さ/境界処理が必要）
                // ConvolveInPlace(payload, f.Width, f.Height, s.Kernel, s.ParamA);

                await _txCh.Writer.WriteAsync(new TxJob(payload, f.Length, f.Timestamp), ct);
            }
            catch
            {
                pool.Return(payload);
                throw;
            }
            finally
            {
                // 入力バッファは返却
                pool.Return(f.Buffer);
            }
        }

        // ====== 例: LUT適用（Mono8/等倍）======
        private static void ApplyLutMono8(byte[] src, byte[] dst, int length, byte[] lut)
        {
            // 単純なループ（必要ならSIMD化）
            for (int i = 0; i < length; i++)
                dst[i] = lut[src[i]];
        }
    }

    // ========= エントリポイント（デモ起動） =========
    class Program
    {
        static void Main(string[] args)
        {
            var usb = new DummyUsbTransport();
            var pipe = new Pipeline(usb, captureCapacity: 8, txCapacity: 32,
                                    degree: Math.Max(1, Environment.ProcessorCount - 1),
                                    capFullMode: BoundedChannelFullMode.DropOldest);

            // デモ用：ダミー eBUS 取得（640x480, 60fps）
            var stream = new PvStream();
            pipe.StartCaptureLoop(stream, width: 640, height: 480, mockFps: 60);
            pipe.StartProcessingWorkers();
            pipe.StartUsbTxLoop();

            Console.WriteLine("Running... Press ENTER to stop.");
            Console.ReadLine();

            pipe.Stop();
            pipe.Dispose();
        }
    }
}


// Program.cs  (.NET Framework 4.8)
// NuGet: System.Threading.Channels, System.Buffers
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace Net48_FifoAgg4_Pipeline
{
    // ======== DTO =========
    public sealed class Frame
    {
        public int Seq { get; }               // 1,2,3,…
        public byte[] Buffer { get; }
        public int Length { get; }
        public int Width { get; }
        public int Height { get; }
        public int Stride { get; }
        public long Timestamp { get; }

        public Frame(int seq, byte[] buffer, int length, int w, int h, int stride, long ts)
        {
            Seq = seq;
            Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
            Length = length; Width = w; Height = h; Stride = stride; Timestamp = ts;
        }
    }

    public sealed class TxJob
    {
        public int Seq { get; }               // 1,2,3,…
        public byte[] Payload { get; }
        public int Length { get; }
        public long Timestamp { get; }
        public TxJob(int seq, byte[] payload, int length, long ts)
        {
            Seq = seq;
            Payload = payload ?? throw new ArgumentNullException(nameof(payload));
            Length = length; Timestamp = ts;
        }
    }

    // ======== 初回処理の共有状態（以降Read専用） ========
    public sealed class InitState
    {
        public readonly byte[] Lut;     // 例: Mono8 LUT
        public readonly int ParamA;     // 例: 係数
        public readonly double[] Kernel;// 例: カーネル
        public InitState(byte[] lut, int a, double[] kernel)
        {
            Lut = lut; ParamA = a; Kernel = kernel;
        }
    }

    // ======== USB 抽象 & ダミー ========
    public interface IUsbTransport
    {
        Task SendAsync(byte[] data, int length, CancellationToken ct);
    }

    public sealed class DummyUsbTransport : IUsbTransport
    {
        public Task SendAsync(byte[] data, int length, CancellationToken ct)
        {
            // 実機では WinUSB/libusb/仮想COM 等に置き換え
            Console.WriteLine($"[USB] sent {length} bytes");
            return Task.CompletedTask;
        }
    }

    // ======== eBUS 擬似（実機に置換） ========
    public sealed class PvStream { }
    public sealed class PvBuffer
    {
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Stride { get; private set; }
        public long Timestamp { get; private set; }
        private IntPtr _ptr;
        public PvBuffer(int w, int h)
        {
            Width = w; Height = h; Stride = w; Timestamp = DateTime.UtcNow.Ticks;
            _ptr = IntPtr.Zero;
        }
        public IntPtr GetDataPointer() => _ptr;
    }

    // ======== パイプライン本体 ========
    public sealed class Pipeline : IDisposable
    {
        // --- Channels ---
        private readonly Channel<Frame> _captureCh;
        private readonly Channel<TxJob> _txCh;

        // --- Tasks & Control ---
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private Task _captureTask;
        private Task[] _workerTasks;
        private Task _txTask;
        private volatile bool _captureStopRequested;

        // --- Config ---
        private readonly IUsbTransport _usb;
        private readonly int _procDegree;

        // --- 初回処理ゲート & 共有状態 ---
        private TaskCompletionSource<bool> _initGate = new TaskCompletionSource<bool>();
        private int _initStarted = 0;          // 0=未開始,1=開始済み
        private InitState _state;              // 初回で設定→以降Read専用

        // --- Seq採番 & 厳密順序 集約(4枚平均) ---
        private int _seqGen = 0;               // 1,2,3,...
        private const int AggregateGroupSize = 4;
        private int _nextGroupToSend = 0;      // groupId=0,1,2,... の順
        private readonly Dictionary<int, List<TxJob>> _groups = new Dictionary<int, List<TxJob>>();

        public Pipeline(
            IUsbTransport usb,
            int captureCapacity = 8,
            int txCapacity = 64,
            int? degree = null)
        {
            _usb = usb ?? throw new ArgumentNullException(nameof(usb));
            _procDegree = degree ?? Math.Max(1, Environment.ProcessorCount - 1);

            // 厳密順序にするなら欠番NG → captureは Wait を推奨
            _captureCh = Channel.CreateBounded<Frame>(new BoundedChannelOptions(captureCapacity)
            {
                FullMode = BoundedChannelFullMode.Wait,   // 欠落禁止
                SingleWriter = true,                      // 取得1本
                SingleReader = false                      // ワーカー複数
            });

            _txCh = Channel.CreateBounded<TxJob>(new BoundedChannelOptions(txCapacity)
            {
                FullMode = BoundedChannelFullMode.Wait,   // 送信前で落とさない
                SingleWriter = false,
                SingleReader = true                       // 送信は単一スレッド
            });
        }

        // ========= 1) 取得（実機では eBUS API に差し替え） =========
        public void StartCaptureLoop(PvStream stream, int width, int height, int mockFps = 60)
        {
            _captureTask = Task.Run(async () =>
            {
                var pool = ArrayPool<byte>.Shared;
                var periodMs = Math.Max(1, 1000 / Math.Max(1, mockFps)); // デモ用fps
                try
                {
                    while (!_captureStopRequested)
                    {
                        // 実機: RetrieveBuffer(out PvBuffer pvb, timeout) → Marshal.Copy / unsafe copy → Requeue
                        int stride = width;
                        int bytes = stride * height;
                        byte[] dst = pool.Rent(bytes);
                        // デモ: ダミーデータ作成
                        for (int i = 0; i < bytes; i++) dst[i] = (byte)(i & 0xFF);
                        long ts = DateTime.UtcNow.Ticks;

                        int seq = Interlocked.Increment(ref _seqGen);
                        await _captureCh.Writer.WriteAsync(new Frame(seq, dst, bytes, width, height, stride, ts), _cts.Token);

                        await Task.Delay(periodMs, _cts.Token);
                    }
                }
                catch (OperationCanceledException) { /* 急停止 */ }
                finally
                {
                    // 新規投入終了
                    _captureCh.Writer.TryComplete();
                }
            }, _cts.Token);
        }

        // ========= 2) 画像処理ワーカー（初回→ゲート→通常） =========
        public void StartProcessingWorkers()
        {
            _workerTasks = new Task[_procDegree];
            for (int i = 0; i < _procDegree; i++)
            {
                _workerTasks[i] = Task.Run(async () =>
                {
                    var pool = ArrayPool<byte>.Shared;

                    try
                    {
                        while (await _captureCh.Reader.WaitToReadAsync(_cts.Token))
                        {
                            Frame f;
                            while (_captureCh.Reader.TryRead(out f))
                            {
                                if (Interlocked.CompareExchange(ref _initStarted, 1, 0) == 0)
                                {
                                    try
                                    {
                                        _state = await DoFirstFrameInitializationAsync(f, _cts.Token); // 初回だけ
                                        _initGate.TrySetResult(true); // 門を開く（可視性保証）
                                    }
                                    catch (Exception ex)
                                    {
                                        _initGate.TrySetException(ex);
                                        throw;
                                    }

                                    await ProcessNormallyUsingStateAsync(f, _cts.Token);
                                }
                                else
                                {
                                    await _initGate.Task;             // 初回完了を待つ
                                    await ProcessNormallyUsingStateAsync(f, _cts.Token);
                                }
                            }
                        }
                    }
                    catch (OperationCanceledException) { /* 急停止 */ }
                }, _cts.Token);
            }
        }

        // ========= 3) 送信（4枚平均 + 厳密順序） =========
        public void StartUsbTxLoop_Aggregate4_Strict()
        {
            _txTask = Task.Run(async () =>
            {
                var pool = ArrayPool<byte>.Shared;
                try
                {
                    _nextGroupToSend = 0;

                    while (await _txCh.Reader.WaitToReadAsync(_cts.Token))
                    {
                        TxJob job;
                        while (_txCh.Reader.TryRead(out job))
                        {
                            int groupId = (job.Seq - 1) / AggregateGroupSize;
                            List<TxJob> list;
                            if (!_groups.TryGetValue(groupId, out list))
                            {
                                list = new List<TxJob>(AggregateGroupSize);
                                _groups[groupId] = list;
                            }
                            list.Add(job);

                            await FlushReadyGroupsInOrderAsync(pool, _cts.Token);
                        }
                    }

                    // close後の最終flush
                    await FlushReadyGroupsInOrderAsync(pool, _cts.Token);

                    // 揃い切らなかったグループを解放（厳密順序なので送らない）
                    foreach (var kv in _groups)
                        foreach (var j in kv.Value) pool.Return(j.Payload);
                    _groups.Clear();
                }
                catch (OperationCanceledException) { /* 急停止 */ }
            }, _cts.Token);
        }

        private async Task FlushReadyGroupsInOrderAsync(ArrayPool<byte> pool, CancellationToken ct)
        {
            List<TxJob> list;
            while (_groups.TryGetValue(_nextGroupToSend, out list))
            {
                if (list.Count < AggregateGroupSize) break; // 揃うまで待つ（厳密順序）

                int len = list[0].Length;                   // 同サイズ前提
                byte[] avg = pool.Rent(len);

                try
                {
                    // 4枚の画素平均（Mono8想定、四捨五入）
                    for (int i = 0; i < len; i++)
                    {
                        int sum = list[0].Payload[i] + list[1].Payload[i] + list[2].Payload[i] + list[3].Payload[i];
                        avg[i] = (byte)((sum + 2) >> 2);
                    }

                    await _usb.SendAsync(avg, len, ct);
                }
                finally
                {
                    // 入力と平均のバッファ返却
                    for (int k = 0; k < AggregateGroupSize; k++)
                        pool.Return(list[k].Payload);
                    pool.Return(avg);
                }

                _groups.Remove(_nextGroupToSend);
                _nextGroupToSend++;
            }
        }

        // ========= 初回処理 & 通常処理（ダミー実装。実処理に差し替えOK） =========
        private Task<InitState> DoFirstFrameInitializationAsync(Frame f, CancellationToken ct)
        {
            // 例：LUT恒等、適当なカーネル
            var lut = new byte[256];
            for (int i = 0; i < lut.Length; i++) lut[i] = (byte)i;
            int a = 42;
            var kernel = new double[] { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
            return Task.FromResult(new InitState(lut, a, kernel));
        }

        private async Task ProcessNormallyUsingStateAsync(Frame f, CancellationToken ct)
        {
            var s = _state; // 可視性OK（_initGate後に参照）
            var pool = ArrayPool<byte>.Shared;
            byte[] payload = pool.Rent(f.Length);
            try
            {
                // 例1: LUT適用（Mono8）
                ApplyLutMono8(f.Buffer, payload, f.Length, s.Lut);
                // 例2: 必要なら畳み込み等もここで（省略）

                await _txCh.Writer.WriteAsync(new TxJob(f.Seq, payload, f.Length, f.Timestamp), ct);
            }
            catch
            {
                pool.Return(payload);
                throw;
            }
            finally
            {
                pool.Return(f.Buffer);
            }
        }

        private static void ApplyLutMono8(byte[] src, byte[] dst, int length, byte[] lut)
        {
            for (int i = 0; i < length; i++) dst[i] = lut[src[i]];
        }

        // ========= 停止系 =========
        /// <summary> 優雅停止：新規取り込みを止め、残りを全て処理＆送信してから止まる </summary>
        public async Task StopGracefullyAsync()
        {
            _captureStopRequested = true;                 // 1) 新規取り込み停止
            if (_captureTask != null) await _captureTask.ConfigureAwait(false); // 2) 取得終了→captureCh close 済み
            if (_workerTasks != null && _workerTasks.Length > 0)
                await Task.WhenAll(_workerTasks).ConfigureAwait(false);         // 3) 全ワーカー完了
            _txCh.Writer.TryComplete();                   // 4) 送信への新規投入停止
            if (_txTask != null) await _txTask.ConfigureAwait(false);           // 5) 送信が読み切って終了
        }

        /// <summary> 強制停止：今すぐ止める（未処理が残る可能性あり） </summary>
        public void Stop() => _cts.Cancel();

        public void Dispose()
        {
            _cts.Cancel();
            _cts.Dispose();
        }
    }

    // ======== エントリポイント ========
    class Program
    {
        static void Main(string[] args)
        {
            var usb = new DummyUsbTransport();
            var pipe = new Pipeline(usb,
                                    captureCapacity: 8,  // 厳密順序なので欠番NG → Wait
                                    txCapacity: 64,
                                    degree: Math.Max(1, Environment.ProcessorCount - 1));

            var stream = new PvStream();
            pipe.StartCaptureLoop(stream, width: 640, height: 480, mockFps: 60);
            pipe.StartProcessingWorkers();
            pipe.StartUsbTxLoop_Aggregate4_Strict();

            Console.WriteLine("Running... ENTER=graceful stop, 'x'+ENTER=force stop");
            var line = Console.ReadLine();
            if (line == "x")
            {
                pipe.Stop();          // 強制停止
            }
            else
            {
                pipe.StopGracefullyAsync().GetAwaiter().GetResult(); // 優雅停止
            }

            pipe.Dispose();
            Console.WriteLine("Stopped.");
        }
    }
}

