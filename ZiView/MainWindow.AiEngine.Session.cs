using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: AI推論セッションのライフサイクル（初期化・実行プロバイダー選択・ウォームアップ）を担当する。
    /// モデル一覧・カテゴリ管理はMainWindow.AiEngine.ModelCatalog.cs、
    /// TensorRTキャッシュ/RAMDISK同期はMainWindow.AiEngine.TrtCache.cs、
    /// タイル分割・バッチ推論本体はMainWindow.AiEngine.Inference.csを参照。
    /// </summary>
    public partial class MainWindow
    {
        private InferenceSession? _onnxSession;
        private string? _inputName;
        private int? _fixedInputSize;
        private bool _isFp16Model = true;
        private string _activeEngineMode = "Unknown";

        // _onnxSession.Run()自体の排他制御用。先読み(Prefetch)スレッドと表示中ページのスレッドが
        // 同じInferenceSession（同じTensorRT実行コンテキスト）に対して同時にRun()を呼ぶと、
        // TensorRTの実行コンテキストは並行呼び出しに対して安全ではないため、.NET側で捕捉できない
        // ネイティブクラッシュ（プロセスごと強制終了）に至ることがある。ページ送りが速いほど
        // 先読みと本表示のタイミングが重なりやすく発生率が上がるため、Run()呼び出し区間だけを直列化する。
        private readonly object _onnxRunLock = new object();

        // InitializeAiの多重同時実行を防ぐフラグ。
        // 「表示時にセッションがnullなら自動再初期化」という既存の保険ロジック（MainWindow.Display.cs）が、
        // 起動直後や名前付きパイプ経由の二重オープン等でTask.Run中のInitializeAiとほぼ同時に走ると、
        // TensorRT/CUDAセッション構築＋ウォームアップが2つ同時にVRAMを取り合い、
        // 実測でVRAM逼迫→OOM（ConvTranspose内でのbad allocation等）→システムメモリ側の
        // 小さなアロケーション（867KB程度）まで失敗する連鎖を引き起こしていた。
        // ロック待ちにはしない（Dispatcher.Invokeを内部で使うためUIスレッドからの呼び出しでデッドロックしうる）。
        // 単純にスキップするだけで、後続のページ表示や自動再init機会で改めて拾われる。
        private volatile bool _aiInitInProgress = false;

        private void InitializeAi() => InitializeAi(_config.SelectedModel);

        /// <summary>
        /// InitializeAi(同期・重い処理)をバックグラウンドスレッドで実行しつつ、
        /// その間は非モーダルのオーバーレイでユーザーに「処理中」であることを示す。
        /// TensorRTエンジン構築・ウォームアップ（初回数十秒かかりうる）でUIが白画面のまま
        /// 応答なしに見える問題への対処。呼び出し中はメインウィンドウの操作を無効化する。
        /// </summary>
        private async Task InitializeAiWithOverlayAsync(string modelFileName)
        {
            var overlay = new EngineStatusOverlay(
                "AIエンジンを構築中です…\n初回はモデル/GPUの組み合わせごとに数十秒かかることがあります。そのままお待ちください。")
            {
                Owner = this
            };
            overlay.Show();
            this.IsEnabled = false;
            try
            {
                await Task.Run(() => InitializeAi(modelFileName));
            }
            finally
            {
                this.IsEnabled = true;
                overlay.Close();

                // オーバーレイ破棄によりOSのフォーカスが外部アプリへ逃げるのを防ぐ
                this.Activate();
                this.Focus();
            }
        }

        private void InitializeAi(string modelFileName)
        {
            if (_aiInitInProgress)
            {
                WriteLog("[AI] InitializeAi already in progress on another thread; skipping duplicate call.");
                return;
            }
            _aiInitInProgress = true;
            try
            {
                InitializeAiCore(modelFileName);
            }
            finally
            {
                _aiInitInProgress = false;
            }
        }

        private void InitializeAiCore(string modelFileName)
        {
            try
            {
                // 既にキャッシュがあれば内部で即return（idempotent）なので、呼び出し元がLoaded/
                // 表示時の自動再init/モデル切替/設定変更のいずれであっても、SSD退避キャッシュからの
                // 復元漏れが起きないようにする）
                RestoreTrtCacheFromBackupIfNeeded();

                string modelPath = Path.Combine(GetModelDirectory(_config.ModelFolder), modelFileName);
                if (!File.Exists(modelPath))
                {
                    WriteLog("ERROR: Model file not found at " + modelPath);
                    Dispatcher.Invoke(() => StatusText.Text = "Mode: Model Missing");
                    return;
                }

                var options = new SessionOptions();
                var available = OrtEnv.Instance().GetAvailableProviders();
                WriteLog($"Available Providers: {string.Join(", ", available)}. Preference: {_config.EnginePreference}");

                // 優先モードごとのフォールバック順序:
                //   TensorRT優先: TensorRT → CUDA → OpenVINO → CPU
                //   CUDA優先    : CUDA → OpenVINO → CPU
                //   OpenVINO優先: OpenVINO → CPU
                List<(string Name, string DisplayMode, Action Append)> chain = _config.EnginePreference switch
                {
                    "CUDA" => new()
                    {
                        ("CUDA", "RTX (CUDA)", () => AppendCuda(options)),
                        ("OpenVINO", "Intel (OpenVINO)", () => AppendOpenVino(options)),
                    },
                    "OpenVINO" => new()
                    {
                        ("OpenVINO", "Intel (OpenVINO)", () => AppendOpenVino(options)),
                    },
                    _ => new() // "TensorRT"（既定）
                    {
                        ("TensorRT", "RTX (TensorRT)", () => AppendTensorRt(options)),
                        ("CUDA", "RTX (CUDA)", () => AppendCuda(options)),
                        ("OpenVINO", "Intel (OpenVINO)", () => AppendOpenVino(options)),
                    },
                };

                _activeEngineMode = "CPU Mode";
                _batchInferenceSupported = true; // モデル/エンジンを切り替えたのでバッチ可否を再判定させる
                _fullFrameSupported = true;       // 同様に全画面一括の可否も再判定させる
                foreach (var (name, displayMode, append) in chain)
                {
                    try
                    {
                        WriteLog($"Attempting {name}...");
                        append();
                        _activeEngineMode = displayMode;
                        break;
                    }
                    catch (Exception exProvider)
                    {
                        WriteLog($"{name} down: {exProvider.Message}");
                    }
                }

                try
                {
                    _onnxSession = new InferenceSession(modelPath, options);
                }
                catch (Exception exSession) when (_activeEngineMode != "CPU Mode")
                {
                    // GPUプロバイダー登録は成功したが、実際のセッション構築で失敗（VRAM不足等）。
                    // CPUのみのオプションで再試行し、完全に使用不能になることを防ぐ。
                    WriteLog($"Session construction failed on {_activeEngineMode}: {exSession.Message}. Retrying with CPU only.");
                    _activeEngineMode = "CPU Mode (Fallback)";
                    _onnxSession = new InferenceSession(modelPath, new SessionOptions());
                }

                _inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault();

                // モデルが固定形状(静的shape)の入力を要求するか検出する。
                // 例: Nomos2系は [1,3,256,256] のように高さ/幅が固定されており、
                // タイルの端数サイズをそのまま渡すと InvalidArgument で必ず失敗する。
                _fixedInputSize = null;
                _isFp16Model = true; // メタデータ取得前の既定値（fp32検出できなければ安全側でfp16扱い＝制限をかけない）
                if (_inputName != null)
                {
                    var meta = _onnxSession.InputMetadata[_inputName];
                    var dims = meta.Dimensions;
                    if (dims.Length >= 4 && dims[2] > 0 && dims[3] > 0)
                    {
                        _fixedInputSize = Math.Max(dims[2], dims[3]);
                        WriteLog($"Model requires fixed input shape: {dims[2]}x{dims[3]}");
                    }
                    _isFp16Model = meta.ElementType == typeof(Float16);
                    WriteLog($"Model precision: {(_isFp16Model ? "fp16" : "fp32")}");
                }

                StatusText.Dispatcher.Invoke(() => StatusText.Text = $"Mode: {_activeEngineMode}");
                WriteLog($"Inference Session context bound. Input: {_inputName}");

                // 初回のGPU/TensorRT初期化遅延を起動時に消化しておく
                WarmupEngine();
            }
            catch (Exception ex)
            {
                _activeEngineMode = "Error";
                Dispatcher.Invoke(() => StatusText.Text = "AI Init Error");
                WriteLog($"CRITICAL ENGINE ABEND: {ex}");
            }
        }

        /// <summary>
        /// AIモデル構築直後にダミー画像を用いて1回推論を実行し、
        /// TensorRT / GPU の初回ウォームアップ（VRAM確保・カーネル準備）を事前に消化する。
        /// </summary>
        private void WarmupEngine()
        {
            if (_onnxSession == null || string.IsNullOrEmpty(_inputName)) return;

            int sz = _fixedInputSize ?? (GetTileSizeForModel(_config.SelectedModel) + 16);

            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                // 固定サイズ（または標準272x272）の黒画像を生成して既存のProcessTileを通す
                using var dummyMat = new Mat(sz, sz, MatType.CV_8UC3, Scalar.All(0));
                using var resultMat = ProcessTile(dummyMat);
                WriteLog($"[AI] Engine warmup (batch=1) complete in {sw.Elapsed.TotalMilliseconds:F0}ms.");
            }
            catch (Exception ex)
            {
                WriteLog($"[AI] Engine warmup skipped/failed: {ex.Message}");
                return; // batch=1すら失敗する状況でバッチwarmupを試みても無意味
            }

            // TensorRT等はバッチサイズごとに別の最適化プロファイルを初回コンパイルするため、
            // 本番で使うバッチサイズ(EffectiveTileBatchSize)もここで一度流して事前コンパイルさせておく。
            // これを怠ると、実ページの初回表示でこのコンパイル待ち（数十秒）がそのままUIフリーズとして
            // 表面化する（実測で発生していた問題）。固定形状モデルはバッチ非対応の可能性が高いため対象外。
            if (!_config.EnableTileBatching)
            {
                WriteLog("[AI] Tile batching disabled by config; using per-tile inference (batch warmup skipped).");
            }
            else if (_fixedInputSize.HasValue)
            {
                // 固定形状モデルのため対象外（何もしない）
            }
            else
            {
                var dummies = new List<Mat>();
                try
                {
                    var swBatch = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < EffectiveTileBatchSize; i++)
                        dummies.Add(new Mat(sz, sz, MatType.CV_8UC3, Scalar.All(0)));

                    var results = ProcessTileBatch(dummies);
                    foreach (var r in results) r.Dispose();
                    WriteLog($"[AI] Engine warmup (batch={EffectiveTileBatchSize}) complete in {swBatch.Elapsed.TotalMilliseconds:F0}ms.");
                }
                catch (Exception exBatch)
                {
                    _batchInferenceSupported = false;
                    WriteLog($"[AI] Batch warmup (batch={EffectiveTileBatchSize}) failed ({exBatch.Message}). " +
                             "Batch inference disabled for this session; using per-tile processing instead.");
                }
                finally
                {
                    foreach (var d in dummies) d.Dispose();
                }
            }

            // 全画面一括推論も、実ページで初めて試すと失敗時に30〜90秒級のコストが表面化するため、
            // 起動時（オーバーレイ表示中）に代表的なサイズで一度試しておく。ここで失敗するモデル/環境は
            // 実運用でもほぼ確実に失敗するため、_fullFrameSupportedをここで確定させて実ページでの
            // 無駄な再試行を防ぐ。固定形状モデルは対象外（そもそも全画面を受け付けない）。
            // 全画面一括推論も、実ページで初めて試すと失敗時に30〜90秒級のコストが表面化するため、
            // 起動時（オーバーレイ表示中）に代表的なサイズで一度試しておく。ここで失敗するモデル/環境は
            // 実運用でもほぼ確実に失敗するため、_fullFrameSupportedをここで確定させて実ページでの
            // 無駄な再試行を防ぐ。固定形状モデルは対象外（そもそも全画面を受け付けない）。
            //
            // ただしこの検証自体（特に失敗パターン）がTensorRTのエンジンビルド試行込みで数十〜90秒級かかるため、
            // 結果をtrt_cacheディレクトリ内にマーカーファイルとして記録し、起動・モデル切替の度に
            // 同じ検証をやり直さないようにする（マーカーはRAMDISK⇔SSD同期の対象フォルダ内に置くため、
            // PC再起動でRAMDISKが消えてもSSD側から一緒に復元される）。
            if (!_fixedInputSize.HasValue)
            {
                string markerPath = GetFullFrameMarkerPath();
                string? cached = ReadFullFrameMarker(markerPath);
                if (cached == "unsupported")
                {
                    _fullFrameSupported = false;
                    WriteLog("[AI] Full-frame support: cached result = unsupported (skipping re-verification).");
                }
                else if (cached == "supported")
                {
                    _fullFrameSupported = true;
                    WriteLog("[AI] Full-frame support: cached result = supported (skipping re-verification).");
                }
                else
                {
                    try
                    {
                        var swFull = System.Diagnostics.Stopwatch.StartNew();
                        using var dummyFull = new Mat(1200, 1688, MatType.CV_8UC3, Scalar.All(0)); // 代表的なコミックページサイズ
                        using var resultFull = ProcessTile(dummyFull);
                        WriteLog($"[AI] Full-frame warmup succeeded in {swFull.Elapsed.TotalMilliseconds:F0}ms. Full-frame inference enabled.");
                        WriteFullFrameMarker(markerPath, "supported");
                    }
                    catch (Exception exFull)
                    {
                        _fullFrameSupported = false;
                        WriteLog($"[AI] Full-frame warmup failed ({exFull.Message}). " +
                                 "Full-frame inference disabled for this session; using tiled inference only.");
                        WriteFullFrameMarker(markerPath, "unsupported");
                    }
                }
            }
        }

        private void AppendTensorRt(SessionOptions options)
        {
            if (_config.TensorRtEngineCacheEnabled)
            {
                string cacheDir = GetTensorRtCacheDirectory(_config.TensorRtCacheDirectory);
                Directory.CreateDirectory(cacheDir);

                var trtOptions = new OrtTensorRTProviderOptions();
                trtOptions.UpdateOptions(new Dictionary<string, string>
                {
                    ["device_id"] = "0",
                    ["trt_engine_cache_enable"] = "1",
                    ["trt_engine_cache_path"] = cacheDir,
                    ["trt_timing_cache_enable"] = "1",

                    // ワークスペース領域の上限を 1GB (1073741824 bytes) や 512MB に絞る 1GBに絞ってみる
                    ["trt_max_workspace_size"] = "1073741824",
                });
                options.AppendExecutionProvider_Tensorrt(trtOptions);
                WriteLog($"TensorRT engine cache enabled: {cacheDir}");
            }
            else
            {
                options.AppendExecutionProvider_Tensorrt(0);
            }
        }

        private void AppendCuda(SessionOptions options)
        {
            options.AppendExecutionProvider_CUDA(0);
        }

        /// <summary>
        /// Intel CPU（および対応環境ではiGPU/NPU）向けのOpenVINO実行プロバイダーを追加する。
        /// 注意: 現行の Microsoft.ML.OnnxRuntime.Gpu パッケージにはOpenVINO用のネイティブライブラリは
        /// 含まれていない。Intel配布のOpenVINO対応ONNX Runtimeパッケージ（またはOpenVINO Runtime本体の
        /// 該当DLL）を別途用意しない環境では、ここで例外となり自動的にCPUへフォールバックする
        /// （TensorRT/CUDAが未導入環境で自動的にフォールバックするのと同じ挙動）。
        /// AppendExecutionProvider_OpenVINOはDictionaryではなくstring（デバイス指定）を取る点に注意。
        /// </summary>
        private void AppendOpenVino(SessionOptions options)
        {
            options.AppendExecutionProvider_OpenVINO("GPU");
        }

        /// <summary>
        /// 現在の推論セッションで実際に使用されているアクティブなVRAM量（MB）の概算値を返す。
        /// </summary>
        public double GetActiveVramUsageMb()
        {
            if (_onnxSession == null) return 0.0;

            try
            {
                // 1. モデルファイル自体のサイズ（VRAM上にロードされている重み）
                double modelWeightMb = 0.0;
                string modelPath = Path.Combine(GetModelDirectory(_config.ModelFolder), _config.SelectedModel);
                if (File.Exists(modelPath))
                {
                    modelWeightMb = new FileInfo(modelPath).Length / (1024.0 * 1024.0);
                }

                // 2. 入出力テンソルバッファの計算
                int elementSize = _isFp16Model ? 2 : 4;
                int tileSize = _fixedInputSize ?? GetTileSizeForModel(_config.SelectedModel);
                int batchSize = _config.EnableTileBatching ? EffectiveTileBatchSize : 1;

                // 入力: [Batch, 3, TileSize, TileSize]
                long inputBytes = (long)batchSize * 3 * tileSize * tileSize * elementSize;
                // 出力: 4倍超解像モデルとして [Batch, 3, TileSize*4, TileSize*4]
                long outputBytes = (long)batchSize * 3 * (tileSize * 4) * (tileSize * 4) * elementSize;

                double tensorMb = (inputBytes + outputBytes) / (1024.0 * 1024.0);

                // 重み + テンソル + 中間特徴マップ作業領域（テンソルの約3倍と仮定）
                return modelWeightMb + (tensorMb * 3.0);
            }
            catch
            {
                return 0.0;
            }
        }
    }
}
