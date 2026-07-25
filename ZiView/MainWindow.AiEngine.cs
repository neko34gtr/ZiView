using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: AIモデルの検出・選択・除外、および
    /// ONNX Runtimeを用いたタイル分割推論（PerformAiTiled/ProcessTile）を担当する。
    /// </summary>
    public partial class MainWindow
    {
        // AI・推論関連のメンバ
        private InferenceSession? _onnxSession;
        private string? _inputName;
        private int? _fixedInputSize;
        private bool _isUpdatingModelComboInternal = false;
        private string _activeEngineMode = "Unknown";

        // 全画面一括推論の適用条件（4K以下・空きVRAM6GB以上ならタイル分割を回避）
        private const int FullFrameMaxLongSidePx = 3840;
        private const long FullFrameMinFreeVramMB = 6000;
        // タイルバッチ推論の1回あたり最大タイル数（TensorRT Dynamic Batch Opt=35を想定）
        private const int TileBatchSize = 35;

        // バッチ推論がモデル/エンジン側で拒否された場合（固定バッチ=1のTensorRTエンジン等）、
        // そのセッション中は繰り返し失敗させず1タイルずつの処理へ切り替える
        private bool _batchInferenceSupported = true;

        // nvidia-smi呼び出しは数十msかかるため、直近の空きVRAM値を一定時間キャッシュして使い回す
        private long _lastFreeVramMB = -1;
        private readonly System.Diagnostics.Stopwatch _vramCheckStopwatch = System.Diagnostics.Stopwatch.StartNew();

        [ThreadStatic] private static float[]? _tileInputBuffer;

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
            }
        }

        private void InitializeAi(string modelFileName)
        {
            try
            {
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
                if (_inputName != null)
                {
                    var meta = _onnxSession.InputMetadata[_inputName];
                    var dims = meta.Dimensions;
                    if (dims.Length >= 4 && dims[2] > 0 && dims[3] > 0)
                    {
                        _fixedInputSize = Math.Max(dims[2], dims[3]);
                        WriteLog($"Model requires fixed input shape: {dims[2]}x{dims[3]}");
                    }
                    bool isFp16 = meta.ElementType == typeof(Float16);
                    WriteLog($"Model precision: {(isFp16 ? "fp16" : "fp32")}");
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
            // 本番で使うバッチサイズ(TileBatchSize)もここで一度流して事前コンパイルさせておく。
            // これを怠ると、実ページの初回表示でこのコンパイル待ち（数十秒）がそのままUIフリーズとして
            // 表面化する（実測で発生していた問題）。固定形状モデルはバッチ非対応の可能性が高いため対象外。
            if (_fixedInputSize.HasValue) return;

            var dummies = new List<Mat>();
            try
            {
                var swBatch = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < TileBatchSize; i++)
                    dummies.Add(new Mat(sz, sz, MatType.CV_8UC3, Scalar.All(0)));

                var results = ProcessTileBatch(dummies);
                foreach (var r in results) r.Dispose();
                WriteLog($"[AI] Engine warmup (batch={TileBatchSize}) complete in {swBatch.Elapsed.TotalMilliseconds:F0}ms.");
            }
            catch (Exception exBatch)
            {
                _batchInferenceSupported = false;
                WriteLog($"[AI] Batch warmup (batch={TileBatchSize}) failed ({exBatch.Message}). " +
                         "Batch inference disabled for this session; using per-tile processing instead.");
            }
            finally
            {
                foreach (var d in dummies) d.Dispose();
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
            options.AppendExecutionProvider_OpenVINO("CPU");
        }

        // 既知モデルのファイル名 → 特性・用途カテゴリの対応表。
        // 未知の *.onnx が追加された場合は GetModelCategory 内のキーワード推定でフォールバックする。
        private static readonly Dictionary<string, string> _modelCategoryMap = new(StringComparer.OrdinalIgnoreCase)
        {
            { "RealESRGAN_x4plus_anime_6B.onnx", "アニメ・コミック" },
            { "4x_cugan_pretrain.onnx", "漫画・イラスト（線画）" },
            { "4x-UltraSharpV2_fp32_op17.onnx", "汎用（実写/イラスト）" },
            { "4xRealWebPhoto_v4_drct-l_fp32.onnx", "実写特化" },
            { "4xNomos2_realplksr_dysample_256_fp32_fullyoptimized.onnx", "漫画・アニメ（高精細）" },
            { "1xDenoise_realplksr_otf_fp32.onnx", "ノイズ除去" },
        };

        /// <summary>
        /// ユーザーが設定ウィンドウで手動割り当てしたカテゴリ（あれば）を優先し、
        /// 無ければ既定の自動分類(GetModelCategory)にフォールバックする。
        /// </summary>
        internal static string GetEffectiveModelCategory(AppConfig config, string fileName)
        {
            if (config.ModelCategoryOverrides.TryGetValue(fileName, out var overridden) && !string.IsNullOrWhiteSpace(overridden))
                return overridden;
            return GetModelCategory(fileName);
        }

        /// <summary>
        /// 設定された ModelFolder を絶対パスへ解決する。
        /// 空文字はプログラムルート（従来互換）、相対パスはルートからの相対、絶対パスはそのまま使用する。
        /// </summary>
        internal static string GetModelDirectory(string configuredFolder)
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            if (string.IsNullOrWhiteSpace(configuredFolder)) return root;
            return Path.IsPathRooted(configuredFolder) ? configuredFolder : Path.Combine(root, configuredFolder);
        }

        /// <summary>
        /// TensorRTエンジンキャッシュの保存先。プログラムルート直下の固定フォルダとし、
        /// ModelFolderの変更に影響されないようにする。
        /// </summary>
        /// <summary>
        /// TensorRTエンジンキャッシュの出力先を決定する。
        /// customDirが指定されていればそれを最優先（書き込み確認込み）。
        /// 未指定時はX:\temp\ZView\trt_cache（RAMDISK運用想定）が使えればそちらへ、
        /// 使えなければプログラムルート直下\trt_cache へフォールバックする。
        /// </summary>
        internal static string GetTensorRtCacheDirectory(string? customDir)
        {
            if (!string.IsNullOrWhiteSpace(customDir))
            {
                try
                {
                    Directory.CreateDirectory(customDir);
                    return customDir;
                }
                catch
                {
                    // 指定フォルダに書き込めない場合は自動判定へフォールバックする
                }
            }

            try
            {
                if (Directory.Exists(@"X:\"))
                {
                    const string ramdiskCacheDir = @"X:\temp\ZView\trt_cache";
                    Directory.CreateDirectory(ramdiskCacheDir);
                    return ramdiskCacheDir;
                }
            }
            catch { }

            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "trt_cache");
        }

        internal static string GetModelCategory(string fileName)
        {
            if (_modelCategoryMap.TryGetValue(fileName, out var cat)) return cat;

            // 未知のモデル向けフォールバック（ファイル名からの推定。確実な分類ではない）
            string lower = fileName.ToLowerInvariant();
            if (lower.Contains("denoise")) return "ノイズ除去";
            if (lower.Contains("cugan")) return "漫画・イラスト（線画）";
            if (lower.Contains("webphoto") || lower.Contains("photo")) return "実写特化";
            if (lower.Contains("nomos")) return "漫画・アニメ（高精細）";
            if (lower.Contains("anime") || lower.Contains("comic")) return "アニメ・コミック";
            if (lower.Contains("sharp") || lower.Contains("general")) return "汎用（実写/イラスト）";
            return "その他";
        }

        /// <summary>
        /// Transformer/Attention系アーキテクチャ（DRCT等）はタイルサイズが大きいとVRAM消費・処理時間が
        /// 急増しやすい傾向があるため、既知の重量級モデルは既定タイルサイズを予防的に下げる。
        /// （未検証の予防的措置であり、確実な不具合修正ではない）
        /// </summary>
        private static int GetTileSizeForModel(string fileName)
        {
            string lower = fileName.ToLowerInvariant();
            if (lower.Contains("drct") || lower.Contains("nomos2")) return 192;
            return 256;
        }

        private Dictionary<string, List<string>> _modelsByCategory = new();

        /// <summary>
        /// プログラムルート直下の *.onnx ファイルを列挙し、ModelComboBoxへ反映する。
        /// 設定に保存済みのモデル名が存在すればそれを、無ければ先頭のモデルを選択状態にする。
        /// 除外リスト(_config.ExcludedModels)に含まれるモデルは一覧から除かれる。
        /// </summary>
        private void ScanOnnxModels()
        {
            try
            {
                string root = GetModelDirectory(_config.ModelFolder);
                if (!Directory.Exists(root))
                {
                    try
                    {
                        Directory.CreateDirectory(root);
                        WriteLog($"Model folder did not exist, created: {root}");
                    }
                    catch (Exception exCreate)
                    {
                        WriteLog($"Model folder creation failed: {exCreate.Message}");
                    }
                }

                var files = Directory.Exists(root)
                    ? Directory.GetFiles(root, "*.onnx")
                        .Select(Path.GetFileName)
                        .Where(f => !string.IsNullOrEmpty(f))
                        .Select(f => f!)
                        .Where(f => !_config.ExcludedModels.Contains(f))
                        .OrderBy(f => f)
                        .ToList()
                    : new List<string>();

                if (files.Count == 0)
                {
                    WriteLog($"ERROR: No usable .onnx model files found in '{root}' (all excluded or missing).");
                    StatusText.Text = "Mode: Model Missing";
                    CategoryComboBox.ItemsSource = null;
                    ModelComboBox.ItemsSource = null;
                    return;
                }

                _modelsByCategory = files
                    .GroupBy(f => GetEffectiveModelCategory(_config, f))
                    .OrderBy(g => g.Key == "その他" ? 1 : 0)
                    .ThenBy(g => g.Key)
                    .ToDictionary(g => g.Key, g => g.OrderBy(x => x).ToList());

                // 保存済みモデルが属するカテゴリを特定し、無ければ先頭カテゴリを初期選択
                string targetCategory = _modelsByCategory
                    .FirstOrDefault(kv => kv.Value.Contains(_config.SelectedModel)).Key
                    ?? _modelsByCategory.Keys.First();

                string targetModel = _modelsByCategory[targetCategory].Contains(_config.SelectedModel)
                    ? _config.SelectedModel
                    : _modelsByCategory[targetCategory][0];

                _isUpdatingModelComboInternal = true;
                CategoryComboBox.ItemsSource = _modelsByCategory.Keys.ToList();
                CategoryComboBox.SelectedItem = targetCategory;
                ModelComboBox.ItemsSource = _modelsByCategory[targetCategory];
                ModelComboBox.SelectedItem = targetModel;
                _isUpdatingModelComboInternal = false;

                _config.SelectedModel = targetModel;
            }
            catch (Exception ex) { WriteLog($"Model Scan Error: {ex.Message}"); }
        }

        private void CategoryComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (_isUpdatingModelComboInternal) return;
            if (CategoryComboBox.SelectedItem is not string category) return;
            if (!_modelsByCategory.TryGetValue(category, out var models) || models.Count == 0) return;

            // カテゴリ変更時は、そのカテゴリ内の先頭モデルを自動選択する
            _isUpdatingModelComboInternal = true;
            ModelComboBox.ItemsSource = models;
            ModelComboBox.SelectedItem = models[0];
            _isUpdatingModelComboInternal = false;

            _ = ApplyModelSelectionAsync(models[0]);
        }

        private async void ModelComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (_isUpdatingModelComboInternal) return;
            if (ModelComboBox.SelectedItem is not string modelFileName) return;
            await ApplyModelSelectionAsync(modelFileName);
        }

        private async Task ApplyModelSelectionAsync(string modelFileName)
        {
            if (modelFileName == _config.SelectedModel && _onnxSession != null) return;

            // 実行中の推論を止め、_onnxSessionへのアクセスが完全に終わるまで待つ
            // （待機自体はawaitのためUIスレッドはブロックされない）
            _cts?.Cancel();
            ClearPageCache(); // 先読みキャッシュはモデル切替前のAI推論結果を保持しているため無効化する
            var runningTask = _currentInferenceTask;
            if (runningTask != null)
            {
                try { await runningTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }

            _config.SelectedModel = modelFileName;

            _onnxSession?.Dispose();
            _onnxSession = null;
            _inputName = null;

            StatusText.Text = "Mode: Loading model...";
            await InitializeAiWithOverlayAsync(modelFileName);

            // モデル切替後、表示中ページをAI再変換
            if (_imageList.Count > 0)
            {
                RefreshDisplay();
            }
        }

        /// <summary>
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
                    WriteLog($"[AI] Full-frame inference failed ({exFull.Message}). Falling back to tiled inference.");
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

            // 残り全タイルの座標だけを先に列挙し（Matはまだ作らない）、TileBatchSize件ずつにまとめて
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

            for (int batchStart = 0; batchStart < tileDescs.Count; batchStart += TileBatchSize)
            {
                token.ThrowIfCancellationRequested();
                int batchCount = Math.Min(TileBatchSize, tileDescs.Count - batchStart);

                var paddedTiles = new List<Mat>(batchCount);
                for (int i = 0; i < batchCount; i++)
                {
                    var (x, y, cw, ch) = tileDescs[batchStart + i];
                    using var tile = new Mat(input, new OpenCvSharp.Rect(x, y, cw, ch));
                    paddedTiles.Add(PadToTarget(tile, targetWAll, targetHAll));
                }

                swBatch.Restart();
                List<Mat> results;
                if (_batchInferenceSupported && batchCount > 1)
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
            WriteLog($"[AI] All {totalTiles} tiles done (batchSize={TileBatchSize}). Output: {output.Width}x{output.Height}");
            return output;
        }

        /// <summary>
        /// 同一サイズにパディング済みのタイル群を1つのバッチテンソル[N,3,H,W]にまとめ、
        /// Session.Runを1回だけ呼び出す。カーネル起動オーバーヘッドをタイル数→バッチ数に削減する。
        /// TensorRT利用時は、エンジン構築時のOptimization ProfileがDynamic Batch
        /// （Min=1, Opt=TileBatchSize, Max=64程度）に対応している必要がある。
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

                using var results = _onnxSession!.Run(inputs);
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

            using var results = _onnxSession!.Run(inputs);

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
