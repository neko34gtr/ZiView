using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: タイル分割・バッチ推論（PerformAiTiled/ProcessTileBatch/ProcessTile）と、
    /// 全画面一括推論の可否判定（空きVRAM・画像サイズ）を担当する。
    /// AIモデル・セッション自体の管理はMainWindow.AiEngine.Session.csを参照。
    /// </summary>
    public partial class MainWindow
    {
        // 全画面一括推論の適用条件（4K以下・空きVRAM6GB以上ならタイル分割を回避）
        // ※画像1枚分の入出力テンソル＋中間特徴マップの実所要量は軽量モデルなら数百MB〜1GB程度。
        //   6GBはTensorRTコンテキスト常駐分（数GB）を考慮しない過剰に安全側の値だったため引き下げた。
        //   万一不足していてもTryFullFrame側のtry/catchでタイル分割へ自動フォールバックするため安全。
        private const int FullFrameMaxLongSidePx = 3840;
        private const long FullFrameMinFreeVramMB = 6144;
        // タイルバッチ推論の1回あたり最大タイル数。
        // ユーザー設定値(AppConfig.TileBatchSize、既定35, 1〜64)を基本としつつ、
        // fp32モデル（fp16の約2倍のメモリを使い、VRAM逼迫→システムメモリへのスワップで
        // 実機がハングした実例がある）を検出した場合は自動的に8以下へ制限する。
        // ユーザーがスライダー等で明示的に小さい値を選んでいればそちらを優先する（Minを取るだけなので下げる分には常に反映される）。
        private int EffectiveTileBatchSize
        {
            get
            {
                int configured = Math.Clamp(_config.TileBatchSize, 1, 64);
                return _isFp16Model ? configured : Math.Min(configured, 8);
            }
        }

        // バッチ推論がモデル/エンジン側で拒否された場合（固定バッチ=1のTensorRTエンジン等）、
        // そのセッション中は繰り返し失敗させず1タイルずつの処理へ切り替える
        private bool _batchInferenceSupported = true;

        // 全画面一括推論が一度でも失敗（形状推論非対応等）したら、そのセッション中は再試行しない。
        // 失敗のたびに30〜90秒以上のエンジンビルド失敗コストを毎ページ払い続けるのを防ぐため。
        private bool _fullFrameSupported = true;

        // nvidia-smi呼び出しは数十msかかるため、直近の空きVRAM値を一定時間キャッシュして使い回す
        private long _lastFreeVramMB = -1;
        private readonly System.Diagnostics.Stopwatch _vramCheckStopwatch = System.Diagnostics.Stopwatch.StartNew();

        [ThreadStatic] private static float[]? _tileInputBuffer;

        /// タイルの入力サイズを、モデルが要求する安全なサイズへ切り上げる。
        /// ・固定形状(_fixedInputSize)を要求するモデルは、必ずそのサイズちょうどに合わせる。
        /// ・可変形状のモデルでも、内部のskip connection等が特定倍数を要求することが多いため、
        /// 　8の倍数へ切り上げることで端数由来の次元不一致（Add演算エラー等）を予防する。
        /// </summary>
        private int GetPaddedTargetSize(int cropSize)
        {
            if (_fixedInputSize.HasValue) return _fixedInputSize.Value;
            const int alignment = 8;
            return ((cropSize + alignment - 1) / alignment) * alignment;
        }

        /// <summary>
        /// 右端・下端のみを鏡映（reflect）でパディングし、指定サイズちょうどに揃える。
        /// 左上を基準に保つことで、後段の配置座標計算をそのまま流用できる。
        /// </summary>
        private static Mat PadToTarget(Mat src, int targetW, int targetH)
        {
            // 通常はtileSize側で防止済みだが、万一クロップが目標サイズを超えていた場合の安全弁
            int cropW = Math.Min(src.Width, targetW);
            int cropH = Math.Min(src.Height, targetH);
            bool needsCrop = cropW != src.Width || cropH != src.Height;
            Mat baseMat = needsCrop ? new Mat(src, new OpenCvSharp.Rect(0, 0, cropW, cropH)) : src;

            int padRight = Math.Max(0, targetW - baseMat.Width);
            int padBottom = Math.Max(0, targetH - baseMat.Height);

            Mat result;
            if (padRight == 0 && padBottom == 0)
            {
                result = baseMat.Clone();
            }
            else
            {
                result = new Mat();
                Cv2.CopyMakeBorder(baseMat, result, 0, padBottom, 0, padRight, BorderTypes.Reflect101);
            }
            if (needsCrop) baseMat.Dispose();
            return result;
        }

        /// <summary>
        /// nvidia-smiを叩いて現在の空きVRAM(MB)を取得する。3秒間キャッシュして呼び出しコストを抑える。
        /// nvidia-smiが無い環境（Intel/AMD/OpenVINO運用等）では-1（不明）を返し、呼び出し元は
        /// 「判定不能＝安全側（一括推論を見送りタイル分割）」として扱う。
        /// </summary>
        private long GetFreeVramMB()
        {
            if (_lastFreeVramMB >= 0 && _vramCheckStopwatch.Elapsed.TotalSeconds < 3)
                return _lastFreeVramMB;

            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    Arguments = "--query-gpu=memory.free --format=csv,noheader,nounits",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var proc = System.Diagnostics.Process.Start(psi);
                string output = proc!.StandardOutput.ReadToEnd();
                proc.WaitForExit(1000);
                string firstLine = output.Split('\n')[0].Trim();
                if (long.TryParse(firstLine, out long freeMb))
                {
                    _lastFreeVramMB = freeMb;
                    _vramCheckStopwatch.Restart();
                    return freeMb;
                }
            }
            catch (Exception ex)
            {
                WriteLog($"[AI] VRAM query unavailable (non-NVIDIA environment?): {ex.Message}");
            }

            _lastFreeVramMB = -1;
            _vramCheckStopwatch.Restart();
            return -1;
        }

        /// <summary>
        /// 全画面一括推論（タイル分割なし）を試みてよい条件かを判定する。
        /// ・固定形状モデルは全画面入力を受け付けられないため対象外
        /// ・CPU Modeでの巨大一括推論はかえって遅くなりやすいため対象外
        /// ・長辺が閾値超、または空きVRAMが閾値未満（判定不能時は安全側でタイル分割）なら対象外
        /// </summary>
        private bool TryGetFullFrameEligible(Mat input, out string reason)
        {
            if (!_fullFrameSupported) { reason = "disabled after earlier failure this session"; return false; }
            if (_fixedInputSize.HasValue) { reason = "fixed-shape model"; return false; }
            if (_activeEngineMode.StartsWith("CPU")) { reason = "CPU mode"; return false; }

            int longSide = Math.Max(input.Width, input.Height);
            if (longSide > FullFrameMaxLongSidePx)
            {
                reason = $"image too large ({input.Width}x{input.Height})";
                return false;
            }

            long freeVram = GetFreeVramMB();
            if (freeVram < FullFrameMinFreeVramMB) // -1(不明)も含め安全側でfalse
            {
                reason = freeVram < 0 ? "VRAM unknown" : $"free VRAM low ({freeVram}MB)";
                return false;
            }

            reason = $"{freeVram}MB free VRAM, {input.Width}x{input.Height}";
            return true;
        }

        /// <summary>
        /// _onnxSession.Run()を排他制御付きで実行する。先読みと本表示が同じセッションへ
        /// 同時にRun()するとTensorRT実行コンテキストの競合でネイティブクラッシュしうるため、
        /// このメソッド経由でのみRun()を呼ぶことで直列化する。
        /// </summary>
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunOnnxSession(List<NamedOnnxValue> inputs)
        {
            lock (_onnxRunLock)
            {
                return _onnxSession!.Run(inputs);
            }
        }

        private Mat PerformAiTiled(Mat input, CancellationToken token, IProgress<(int done, int total)>? progress = null)
        {
            // 条件を満たせばタイル分割自体を回避し、画像全体を1回のRunで処理する。
            // 失敗した場合（Dynamic Shape非対応モデル等）は例外を握りタイル分割へフォールバックする。
            if (TryGetFullFrameEligible(input, out string ffReason))
            {
                try
                {
                    var swFull = System.Diagnostics.Stopwatch.StartNew();
                    WriteLog($"[AI] Attempting full-frame inference ({ffReason}).");
                    progress?.Report((0, 1));
                    var fullResult = ProcessTile(input);
                    progress?.Report((1, 1));
                    WriteLog($"[AI] Full-frame inference succeeded: {input.Width}x{input.Height} -> " +
                             $"{fullResult.Width}x{fullResult.Height} in {swFull.Elapsed.TotalMilliseconds:F0}ms.");
                    return fullResult;
                }
                catch (Exception exFull)
                {
                    _fullFrameSupported = false;
                    WriteLog($"[AI] Full-frame inference failed ({exFull.Message}). " +
                             "Disabling full-frame for this session; using tiled inference from now on.");
                }
            }

            else
            {
                WriteLog($"[AI] Full-frame inference skipped ({ffReason}). Using tiled+batched path.");
            }

            int tileSize = GetTileSizeForModel(_config.SelectedModel);
            int overlap = 16;

            if (_fixedInputSize.HasValue)
            {
                // 固定形状モデルは要求サイズより大きいクロップを渡すと必ず失敗するため、
                // tileSize+overlap が要求サイズちょうどになるよう強制的に上書きする
                int fixedSize = _fixedInputSize.Value;
                overlap = Math.Min(overlap, Math.Max(0, fixedSize / 8));
                tileSize = Math.Max(1, fixedSize - overlap);
            }

            int inWidth = input.Width;
            int inHeight = input.Height;

            int tileCountX = (int)Math.Ceiling(inWidth / (double)tileSize);
            int tileCountY = (int)Math.Ceiling(inHeight / (double)tileSize);
            int totalTiles = tileCountX * tileCountY;
            WriteLog($"[AI] Start: {inWidth}x{inHeight}, model={_config.SelectedModel}, tileSize={tileSize}, " +
                     $"fixedInput={(_fixedInputSize?.ToString() ?? "dynamic")}, tiles={totalTiles} ({tileCountX}x{tileCountY})");
            progress?.Report((0, totalTiles));

            // 最初のタイルを実際に推論し、出力テンソルの実寸からスケール倍率を算出する
            // （1x/2x/3x/4x等、モデルごとに異なるため決め打ちにしない）
            var swProbe = System.Diagnostics.Stopwatch.StartNew();
            int standardTileSize = tileSize + overlap; // 入力サイズを常にこのサイズに完全固定化する(実験)
            int probeCw = Math.Min(standardTileSize, inWidth);
            int probeCh = Math.Min(standardTileSize, inHeight);
            int probeTargetW = _fixedInputSize ?? standardTileSize;
            int probeTargetH = _fixedInputSize ?? standardTileSize;

            Mat probeResult;
            using (var probeCrop = new Mat(input, new OpenCvSharp.Rect(0, 0, probeCw, probeCh)))
            using (var probePadded = PadToTarget(probeCrop, probeTargetW, probeTargetH))
            {
                probeResult = ProcessTile(probePadded);
            }
            double scaleX = (double)probeResult.Width / probeTargetW;
            double scaleY = (double)probeResult.Height / probeTargetH;
            WriteLog($"[AI] Probe tile done in {swProbe.Elapsed.TotalMilliseconds:F0}ms. Detected scale: {scaleX:F2}x / {scaleY:F2}x " +
                     $"(padded {probeCw}x{probeCh} -> {probeTargetW}x{probeTargetH})");
            progress?.Report((1, totalTiles));

            int outWidth = (int)Math.Round(inWidth * scaleX);
            int outHeight = (int)Math.Round(inHeight * scaleY);
            Mat output = new Mat(outHeight, outWidth, input.Type());

            // プローブ結果を実データ分だけ切り出して(0,0)へそのまま利用する
            int probeRealW = (int)Math.Round(probeCw * scaleX);
            int probeRealH = (int)Math.Round(probeCh * scaleY);
            using (var probeRoi = new Mat(probeResult, new OpenCvSharp.Rect(0, 0,
                       Math.Min(probeRealW, probeResult.Width), Math.Min(probeRealH, probeResult.Height))))
            using (var dst0 = new Mat(output, new OpenCvSharp.Rect(0, 0, probeRoi.Width, probeRoi.Height)))
            {
                probeRoi.CopyTo(dst0);
            }
            probeResult.Dispose();

            int targetWAll = _fixedInputSize ?? standardTileSize;
            int targetHAll = _fixedInputSize ?? standardTileSize;

            // 残り全タイルの座標だけを先に列挙し（Matはまだ作らない）、EffectiveTileBatchSize件ずつにまとめて
            // 1回のSession.Runへ一括投入する。カーネル起動回数がタイル数→バッチ数に減るのが狙い。
            var tileDescs = new List<(int x, int y, int cw, int ch)>();
            for (int y = 0; y < inHeight; y += tileSize)
            {
                for (int x = 0; x < inWidth; x += tileSize)
                {
                    if (x == 0 && y == 0) continue; // プローブで処理済み
                    int cw = Math.Min(tileSize + overlap, inWidth - x);
                    int ch = Math.Min(tileSize + overlap, inHeight - y);
                    tileDescs.Add((x, y, cw, ch));
                }
            }

            int tileIndex = 1; // プローブ分を1件消化済みとして数える
            var swBatch = System.Diagnostics.Stopwatch.StartNew();
            var swLogThrottle = System.Diagnostics.Stopwatch.StartNew(); // ログの間引き用（一定時間おきのみ出力）

            for (int batchStart = 0; batchStart < tileDescs.Count; batchStart += EffectiveTileBatchSize)
            {
                token.ThrowIfCancellationRequested();
                int batchCount = Math.Min(EffectiveTileBatchSize, tileDescs.Count - batchStart);

                var paddedTiles = new List<Mat>(batchCount);
                for (int i = 0; i < batchCount; i++)
                {
                    var (x, y, cw, ch) = tileDescs[batchStart + i];
                    using var tile = new Mat(input, new OpenCvSharp.Rect(x, y, cw, ch));
                    paddedTiles.Add(PadToTarget(tile, targetWAll, targetHAll));
                }

                swBatch.Restart();
                List<Mat> results;
                if (_config.EnableTileBatching && _batchInferenceSupported && batchCount > 1)
                {
                    try
                    {
                        results = ProcessTileBatch(paddedTiles);
                    }
                    catch (Exception exBatch)
                    {
                        // TensorRTエンジンが静的バッチ=1でビルドされている場合等はここに落ちる。
                        // Dynamic Batch(Min=1,Opt=35,Max=64)非対応と判断し、以降このセッションでは
                        // 1タイルずつのProcessTileへ切り替える（正しさを優先）。
                        _batchInferenceSupported = false;
                        WriteLog($"[AI] Batched inference rejected ({exBatch.Message}). " +
                                 "Falling back to per-tile inference for this session (needs Dynamic Batch profile on the TensorRT engine).");
                        results = paddedTiles.Select(ProcessTile).ToList();
                    }
                }
                else
                {
                    results = paddedTiles.Select(ProcessTile).ToList();
                }
                foreach (var p in paddedTiles) p.Dispose();

                for (int i = 0; i < batchCount; i++)
                {
                    var (x, y, cw, ch) = tileDescs[batchStart + i];
                    var up = results[i];
                    tileIndex++;

                    bool isLastTile = tileIndex == totalTiles;
                    if (isLastTile || swLogThrottle.Elapsed.TotalMilliseconds >= 300)
                    {
                        WriteLog($"[AI] Tile {tileIndex}/{totalTiles} (x={x},y={y},{cw}x{ch}, batch={batchCount}) " +
                                 $"-> {up.Width}x{up.Height} in {swBatch.Elapsed.TotalMilliseconds:F0}ms/batch");
                        swLogThrottle.Restart();
                    }
                    progress?.Report((tileIndex, totalTiles));

                    token.ThrowIfCancellationRequested();

                    // パディング分を除いた「実データ相当」の範囲だけを結果として使う
                    int realOw = (int)Math.Round(cw * scaleX);
                    int realOh = (int)Math.Round(ch * scaleY);
                    int dx = (int)Math.Round(x * scaleX);
                    int dy = (int)Math.Round(y * scaleY);
                    int ow = Math.Min(Math.Min(realOw, up.Width), output.Width - dx);
                    int oh = Math.Min(Math.Min(realOh, up.Height), output.Height - dy);

                    using (var roiSrc = new Mat(up, new OpenCvSharp.Rect(0, 0, ow, oh)))
                    using (var roiDst = new Mat(output, new OpenCvSharp.Rect(dx, dy, ow, oh)))
                    {
                        roiSrc.CopyTo(roiDst);
                    }
                    up.Dispose();
                }
            }
            WriteLog($"[AI] All {totalTiles} tiles done (batchSize={EffectiveTileBatchSize}). Output: {output.Width}x{output.Height}");
            return output;
        }

        /// <summary>
        /// AI推論時のVRAM/RAM暴走・メモリ不足を防ぐため、
        /// 長辺が指定サイズ（デフォルト3840px）を超える場合はアスペクト比を維持して縮小する。
        /// </summary>
        private Mat PrepareMatForInference(Mat src, int maxLongSide = 3840)
        {
            if (src == null || src.Empty()) return new Mat();

            int maxSide = Math.Max(src.Width, src.Height);
            if (maxSide <= maxLongSide)
            {
                return src.Clone();
            }

            double scale = (double)maxLongSide / maxSide;
            int newWidth = (int)(src.Width * scale);
            int newHeight = (int)(src.Height * scale);

            Mat resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(newWidth, newHeight), 0, 0, InterpolationFlags.Area);
            WriteLog($"[AI] Input image downsampled for inference: {src.Width}x{src.Height} -> {newWidth}x{newHeight}");
            return resized;
        }

        /// <summary>
        /// 同一サイズにパディング済みのタイル群を1つのバッチテンソル[N,3,H,W]にまとめ、
        /// Session.Runを1回だけ呼び出す。カーネル起動オーバーヘッドをタイル数→バッチ数に削減する。
        /// TensorRT利用時は、エンジン構築時のOptimization ProfileがDynamic Batch
        /// （Min=1, Opt=EffectiveTileBatchSize, Max=64程度）に対応している必要がある。
        /// 非対応の場合はここで例外になり、呼び出し元がper-tile処理へフォールバックする。
        /// </summary>
        [ThreadStatic] private static float[]? _batchInputBuffer;
        [ThreadStatic] private static Float16[]? _batchInputBufferFp16;

        private List<Mat> ProcessTileBatch(List<Mat> tiles)
        {
            int n = tiles.Count;
            int w = tiles[0].Width, h = tiles[0].Height;
            bool useFp16 = _inputName != null && _onnxSession!.InputMetadata[_inputName].ElementType == typeof(Float16);

            int perTile = 3 * w * h;
            int required = n * perTile;
            List<NamedOnnxValue> inputs;

            // Pinned Memory対応：バッチ用バッファをスレッドごとに使い回し、Run()実行中は
            // GCHandleで明示的にピン留めする。毎バッチ new float[n*perTile]（数十MB）していた
            // アロケーション＆GC負荷を解消し、GCによる再配置リスクも無くす。
            GCHandle pinHandle = default;
            bool pinned = false;
            try
            {
                if (useFp16)
                {
                    if (_batchInputBufferFp16 == null || _batchInputBufferFp16.Length < required)
                        _batchInputBufferFp16 = new Float16[required];
                    var data16 = _batchInputBufferFp16;
                    pinHandle = GCHandle.Alloc(data16, GCHandleType.Pinned);
                    pinned = true;

                    Parallel.For(0, n, ti =>
                    {
                        using var rgb = new Mat();
                        Cv2.CvtColor(tiles[ti], rgb, ColorConversionCodes.BGR2RGB);
                        var idx = rgb.GetUnsafeGenericIndexer<Vec3b>();
                        int baseOff = ti * perTile;
                        for (int y = 0; y < h; y++)
                            for (int x = 0; x < w; x++)
                            {
                                var v = idx[y, x];
                                data16[baseOff + 0 * w * h + y * w + x] = (Float16)(v.Item0 / 255f);
                                data16[baseOff + 1 * w * h + y * w + x] = (Float16)(v.Item1 / 255f);
                                data16[baseOff + 2 * w * h + y * w + x] = (Float16)(v.Item2 / 255f);
                            }
                    });
                    var tensor16 = required == data16.Length
                        ? new DenseTensor<Float16>(data16, new[] { n, 3, h, w })
                        : new DenseTensor<Float16>(new Memory<Float16>(data16, 0, required), new[] { n, 3, h, w });
                    inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName ?? "input", tensor16) };
                }
                else
                {
                    if (_batchInputBuffer == null || _batchInputBuffer.Length < required)
                        _batchInputBuffer = new float[required];
                    var data = _batchInputBuffer;
                    pinHandle = GCHandle.Alloc(data, GCHandleType.Pinned);
                    pinned = true;

                    Parallel.For(0, n, ti =>
                    {
                        using var rgb = new Mat();
                        Cv2.CvtColor(tiles[ti], rgb, ColorConversionCodes.BGR2RGB);
                        var idx = rgb.GetUnsafeGenericIndexer<Vec3b>();
                        int baseOff = ti * perTile;
                        for (int y = 0; y < h; y++)
                            for (int x = 0; x < w; x++)
                            {
                                var v = idx[y, x];
                                data[baseOff + 0 * w * h + y * w + x] = v.Item0 / 255f;
                                data[baseOff + 1 * w * h + y * w + x] = v.Item1 / 255f;
                                data[baseOff + 2 * w * h + y * w + x] = v.Item2 / 255f;
                            }
                    });
                    var tensor = required == data.Length
                        ? new DenseTensor<float>(data, new[] { n, 3, h, w })
                        : new DenseTensor<float>(new Memory<float>(data, 0, required), new[] { n, 3, h, w });
                    inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName ?? "input", tensor) };
                }

                //using var results = _onnxSession!.Run(inputs);
                using var results = RunOnnxSession(inputs);
                return ConvertBatchResults(results, n, useFp16);
            }
            finally
            {
                if (pinned) pinHandle.Free();
            }
        }


        /// <summary>Session.Runの出力テンソルをタイルごとのMatへ分解する（ProcessTileBatchから分離）。</summary>
        private List<Mat> ConvertBatchResults(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, int n, bool useFp16)
        {
            var outList = new List<Mat>(n);

            if (useFp16)
            {
                var output16 = results.First().AsTensor<Float16>();
                int outH = output16.Dimensions[2], outW = output16.Dimensions[3];
                for (int ti = 0; ti < n; ti++)
                {
                    var res = new Mat(outH, outW, MatType.CV_8UC3);
                    var resIdx = res.GetUnsafeGenericIndexer<Vec3b>();
                    int t = ti;
                    Parallel.For(0, outH, y =>
                    {
                        for (int x = 0; x < outW; x++)
                        {
                            resIdx[y, x] = new Vec3b(
                                (byte)Math.Clamp((float)output16[t, 2, y, x] * 255, 0, 255),
                                (byte)Math.Clamp((float)output16[t, 1, y, x] * 255, 0, 255),
                                (byte)Math.Clamp((float)output16[t, 0, y, x] * 255, 0, 255));
                        }
                    });
                    outList.Add(res);
                }
                return outList;
            }

            var output = results.First().AsTensor<float>();
            int oH = output.Dimensions[2], oW = output.Dimensions[3];
            for (int ti = 0; ti < n; ti++)
            {
                var res = new Mat(oH, oW, MatType.CV_8UC3);
                var resIdx = res.GetUnsafeGenericIndexer<Vec3b>();
                int t = ti;
                Parallel.For(0, oH, y =>
                {
                    for (int x = 0; x < oW; x++)
                    {
                        resIdx[y, x] = new Vec3b(
                            (byte)Math.Clamp(output[t, 2, y, x] * 255, 0, 255),
                            (byte)Math.Clamp(output[t, 1, y, x] * 255, 0, 255),
                            (byte)Math.Clamp(output[t, 0, y, x] * 255, 0, 255));
                    }
                });
                outList.Add(res);
            }
            return outList;
        }

        [ThreadStatic] private static Float16[]? _tileInputBufferFp16;

        private Mat ProcessTile(Mat tile)
        {
            int w = tile.Width, h = tile.Height;
            using var rgb = new Mat();
            Cv2.CvtColor(tile, rgb, ColorConversionCodes.BGR2RGB);

            // 入力テンソルの要素型を見て fp16/fp32 を自動判定する（fp16軽量版モデル対応）
            bool useFp16 = _inputName != null && _onnxSession!.InputMetadata[_inputName].ElementType == typeof(Float16);

            int required = 3 * w * h;
            var idx = rgb.GetUnsafeGenericIndexer<Vec3b>();
            List<NamedOnnxValue> inputs;

            if (useFp16)
            {
                if (_tileInputBufferFp16 == null || _tileInputBufferFp16.Length < required)
                {
                    _tileInputBufferFp16 = new Float16[required];
                }
                var data16 = _tileInputBufferFp16;
                // 各yは書き込み先が完全に独立しているため、行単位の並列化で安全に高速化できる
                Parallel.For(0, h, y =>
                {
                    for (int x = 0; x < w; x++)
                    {
                        var v = idx[y, x];
                        data16[0 * w * h + y * w + x] = (Float16)(v.Item0 / 255f);
                        data16[1 * w * h + y * w + x] = (Float16)(v.Item1 / 255f);
                        data16[2 * w * h + y * w + x] = (Float16)(v.Item2 / 255f);
                    }
                });
                var tensor16 = required == data16.Length
                    ? new DenseTensor<Float16>(data16, new[] { 1, 3, h, w })
                    : new DenseTensor<Float16>(new Memory<Float16>(data16, 0, required), new[] { 1, 3, h, w });
                inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName ?? "input", tensor16) };
            }
            else
            {
                if (_tileInputBuffer == null || _tileInputBuffer.Length < required)
                {
                    _tileInputBuffer = new float[required];
                }
                var data = _tileInputBuffer;
                Parallel.For(0, h, y =>
                {
                    for (int x = 0; x < w; x++)
                    {
                        var v = idx[y, x];
                        data[0 * w * h + y * w + x] = v.Item0 / 255f;
                        data[1 * w * h + y * w + x] = v.Item1 / 255f;
                        data[2 * w * h + y * w + x] = v.Item2 / 255f;
                    }
                });
                // バッファを使い回すため、テンソルには実サイズ分だけを渡す（末尾の余剰領域は無視される）
                var tensor = required == data.Length
                    ? new DenseTensor<float>(data, new[] { 1, 3, h, w })
                    : new DenseTensor<float>(new Memory<float>(data, 0, required), new[] { 1, 3, h, w });
                inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName ?? "input", tensor) };
            }

            //using var results = _onnxSession!.Run(inputs);
            using var results = RunOnnxSession(inputs);

            int outH, outW;
            Mat res;

            if (useFp16)
            {
                // fp16モデルは出力も同じくfp16である前提（fp16変換モデルの一般的な仕様）
                var output16 = results.First().AsTensor<Float16>();
                outH = output16.Dimensions[2];
                outW = output16.Dimensions[3];
                res = new Mat(outH, outW, MatType.CV_8UC3);
                var resIdx16 = res.GetUnsafeGenericIndexer<Vec3b>();
                Parallel.For(0, outH, y =>
                {
                    for (int x = 0; x < outW; x++)
                    {
                        resIdx16[y, x] = new Vec3b(
                            (byte)Math.Clamp((float)output16[0, 2, y, x] * 255, 0, 255),
                            (byte)Math.Clamp((float)output16[0, 1, y, x] * 255, 0, 255),
                            (byte)Math.Clamp((float)output16[0, 0, y, x] * 255, 0, 255)
                        );
                    }
                });
                return res;
            }

            var output = results.First().AsTensor<float>();

            // 出力サイズをテンソルの実寸から取得する（モデルのスケール倍率を仮定しない）
            outH = output.Dimensions[2];
            outW = output.Dimensions[3];

            res = new Mat(outH, outW, MatType.CV_8UC3);
            var resIdx = res.GetUnsafeGenericIndexer<Vec3b>();
            Parallel.For(0, outH, y =>
            {
                for (int x = 0; x < outW; x++)
                {
                    resIdx[y, x] = new Vec3b(
                        (byte)Math.Clamp(output[0, 2, y, x] * 255, 0, 255),
                        (byte)Math.Clamp(output[0, 1, y, x] * 255, 0, 255),
                        (byte)Math.Clamp(output[0, 0, y, x] * 255, 0, 255)
                    );
                }
            });
            return res;
        }
    }
}
