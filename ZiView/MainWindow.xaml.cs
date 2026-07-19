using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression; // .NET標準のZIP圧縮マネージャー
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

using ImageMagick;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ZiView
{
    /// <summary>
    /// AppConfigクラス:
    /// アプリケーションの永続的な設定（ウィンドウ位置、チェックボックスの状態、最後に開いたパス等）を管理します。
    /// </summary>
    public class AppConfig
    {
        public double WindowLeft { get; set; } = 100;
        public double WindowTop { get; set; } = 100;
        public double WindowWidth { get; set; } = 1200;
        public double WindowHeight { get; set; } = 800;
        public bool CheckSpread { get; set; } = false;
        public bool CheckAutoDetectOrder { get; set; } = false; // プロパティ名修正等の不整合防止
        public bool CheckAutoDetect { get; set; } = false;
        public bool CheckPrefetch { get; set; } = false;
        public double SplitSliderValue { get; set; } = 100;
        public string LastSourcePath { get; set; } = string.Empty;

        // レンズ補正およびレティクル用永続パラメータ
        public bool ShowReticle { get; set; } = true;
        public bool EnableLensCorrection { get; set; } = false;
        public double LensCorrectionAmount { get; set; } = 0.40;

        // 背景色の設定を保存するプロパティ（デフォルト: 真の黒）
        public string BackgroundColor { get; set; } = "#000000";

        // 選択中のAIモデルファイル名（プログラムルート直下の *.onnx）
        public string SelectedModel { get; set; } = "RealESRGAN_x4plus_anime_6B.onnx";

        // 動作が重すぎる等の理由でユーザーが選択肢から除外したモデル（ファイル名一覧）
        public List<string> ExcludedModels { get; set; } = new();
    }

    public partial class MainWindow : System.Windows.Window
    {
        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);
        private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

        // AI・描画関連のメンバ
        private InferenceSession? _onnxSession;
        private string? _inputName;
        private int? _fixedInputSize;
        private bool _isUpdatingModelComboInternal = false;
        private string _activeEngineMode = "Unknown";

        private string? _currentSourcePath;
        private List<string> _imageList = new List<string>();
        private Mat? _currentCombinedOriginal;
        private Mat? _currentCombinedUpscaled;
        private readonly DispatcherTimer _monitorTimer;

        // パス定義
        private readonly string _configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "zi_view_config.json");
        private readonly string _logPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "session.log");
        private readonly string _tempExtractDir = Path.Combine(Path.GetTempPath(), "ZiView_Temp");

        // UIアニメーション・インタラクション状態管理
        private bool _isSidebarOpen = false;
        private bool _isBottomBarOpen = false;
        private System.Windows.Point _startPoint;
        private System.Windows.Point _origin;
        private bool _isDragging;
        private AppConfig _config = new();
        private CancellationTokenSource? _cts;
        private Task? _currentInferenceTask;

        // 無限ループイベントを抑止するためのフラグ
        private bool _isUpdatingPageSliderInternal = false;

        // カラーピッカー用状態変数
        private double _currentHue = 0.0;
        private double _currentSaturation = 1.0;
        private double _currentValue = 1.0;
        private string _selectedHexColor = "#000000";

        public MainWindow(string[]? args = null)
        {
            InitializeComponent();
            InitLog();
            WriteLog("ZiView Engine Starting (Native Core Standard)...");

            LoadConfig();

            if (args != null && args.Length > 0)
            {
                _currentSourcePath = args[0];
            }

            this.SourceInitialized += (s, e) =>
            {
                IntPtr hwnd = new WindowInteropHelper(this).Handle;
                int darkMode = 1;
                DwmSetWindowAttribute(hwnd, 20, ref darkMode, sizeof(int));
            };

            _monitorTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
            _monitorTimer.Tick += (s, e) => UpdateMemoryUsage();
            _monitorTimer.Start();

            this.Loaded += (s, e) =>
            {
                ScanOnnxModels();
                InitializeAi();
                ApplyConfigToUi();
                if (!string.IsNullOrEmpty(_currentSourcePath))
                {
                    LoadSource(_currentSourcePath);
                }

                // 設定から読み込んだ背景色（_config.BackgroundColor）を反映し、カラーピッカーの状態を同期する
                try
                {
                    var mediaColor = (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString(_config.BackgroundColor);
                    RgbToHsv(mediaColor, out _currentHue, out _currentSaturation, out _currentValue);
                    SliderHue.Value = _currentHue;
                }
                catch { }

                UpdateColorPickerBrush();
                UpdatePreview();
            };
            SetupEvents();
        }
        private void RgbToHsv(System.Windows.Media.Color color, out double h, out double s, out double v)
        {
            double r = color.R / 255.0;
            double g = color.G / 255.0;
            double b = color.B / 255.0;

            double min = Math.Min(Math.Min(r, g), b);
            double max = Math.Max(Math.Max(r, g), b);
            double delta = max - min;

            v = max;
            s = max == 0 ? 0 : delta / max;

            if (delta == 0)
            {
                h = 0;
            }
            else if (max == r)
            {
                h = 60 * ((g - b) / delta % 6);
            }
            else if (max == g)
            {
                h = 60 * ((b - r) / delta + 2);
            }
            else
            {
                h = 60 * ((r - g) / delta + 4);
            }

            if (h < 0) h += 360;
        }

        private void InitLog()
        {
            try { File.WriteAllText(_logPath, $"=== Session Started at {DateTime.Now} ===\n"); } catch { }
        }

        private readonly object _logLock = new();
        private void WriteLog(string message)
        {
            lock (_logLock)
            {
                try { File.AppendAllText(_logPath, $"[{DateTime.Now:HH:mm:ss}] {message}\n"); } catch { }
            }
        }

        private void LoadConfig()
        {
            try
            {
                if (File.Exists(_configPath))
                {
                    string json = File.ReadAllText(_configPath, System.Text.Encoding.UTF8);
                    var loaded = JsonSerializer.Deserialize<AppConfig>(json);
                    if (loaded != null)
                    {
                        _config = loaded;
                        this.Left = _config.WindowLeft;
                        this.Top = _config.WindowTop;
                        this.Width = _config.WindowWidth;
                        this.Height = _config.WindowHeight;
                        _currentSourcePath = _config.LastSourcePath;
                    }
                }
            }
            catch (Exception ex) { WriteLog($"Config Load Error: {ex.Message}"); }
        }

        private void SaveConfig()
        {
            try
            {
                _config.WindowLeft = this.Left;
                _config.WindowTop = this.Top;
                _config.WindowWidth = this.ActualWidth;
                _config.WindowHeight = this.ActualHeight;
                _config.CheckSpread = CheckSpread.IsChecked ?? true;
                _config.CheckAutoDetect = CheckAutoDetect.IsChecked ?? false;
                _config.CheckPrefetch = CheckPrefetch.IsChecked ?? true;
                _config.ShowReticle = CheckReticle.IsChecked ?? true;
                _config.SplitSliderValue = SplitSlider.Value;
                _config.LastSourcePath = _currentSourcePath ?? string.Empty;
                _config.BackgroundColor = _selectedHexColor;
                if (ModelComboBox.SelectedItem is string selectedModel) _config.SelectedModel = selectedModel;

                var options = new JsonSerializerOptions { WriteIndented = true };
                string json = JsonSerializer.Serialize(_config, options);
                File.WriteAllText(_configPath, json, new System.Text.UTF8Encoding(false));
            }
            catch (Exception ex) { WriteLog($"Config Save Error: {ex.Message}"); }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            SaveConfig();
            WriteLog("Cleaning up resources.");
            _onnxSession?.Dispose();
            _currentCombinedOriginal?.Dispose();
            _currentCombinedUpscaled?.Dispose();

            try
            {
                if (Directory.Exists(_tempExtractDir))
                {
                    Directory.Delete(_tempExtractDir, true);
                }
            }
            catch { }
        }

        private void InitializeAi() => InitializeAi(_config.SelectedModel);

        private void InitializeAi(string modelFileName)
        {
            try
            {
                string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, modelFileName);
                if (!File.Exists(modelPath))
                {
                    WriteLog("ERROR: Model file not found at " + modelPath);
                    StatusText.Text = "Mode: Model Missing";
                    return;
                }

                var options = new SessionOptions();
                var available = OrtEnv.Instance().GetAvailableProviders();
                WriteLog($"Available Providers: {string.Join(", ", available)}");

                try
                {
                    WriteLog("Attempting TensorRT...");
                    options.AppendExecutionProvider_Tensorrt(0);
                    _activeEngineMode = "RTX (TensorRT)";
                }
                catch (Exception exTrt)
                {
                    WriteLog($"TensorRT down: {exTrt.Message}");
                    try
                    {
                        WriteLog("Attempting CUDA...");
                        options.AppendExecutionProvider_CUDA(0);
                        _activeEngineMode = "RTX (CUDA)";
                    }
                    catch (Exception exCuda)
                    {
                        WriteLog($"CUDA down: {exCuda.Message}");
                        try
                        {
                            WriteLog("Attempting DirectML...");
                            options.AppendExecutionProvider_DML(0);
                            _activeEngineMode = "DirectML";
                        }
                        catch (Exception ex2)
                        {
                            WriteLog($"DirectML down: {ex2.Message}");
                            _activeEngineMode = "CPU Mode";
                        }
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
                    var dims = _onnxSession.InputMetadata[_inputName].Dimensions;
                    if (dims.Length >= 4 && dims[2] > 0 && dims[3] > 0)
                    {
                        _fixedInputSize = Math.Max(dims[2], dims[3]);
                        WriteLog($"Model requires fixed input shape: {dims[2]}x{dims[3]}");
                    }
                }

                StatusText.Text = $"Mode: {_activeEngineMode}";
                WriteLog($"Inference Session context bound. Input: {_inputName}");
            }
            catch (Exception ex)
            {
                _activeEngineMode = "Error";
                StatusText.Text = $"AI Init Error";
                WriteLog($"CRITICAL ENGINE ABEND: {ex}");
            }
        }

        /// <summary>
        /// プログラムルート直下の *.onnx ファイルを列挙し、ModelComboBoxへ反映する。
        /// 設定に保存済みのモデル名が存在すればそれを、無ければ先頭のモデルを選択状態にする。
        /// </summary>
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

        private static string GetModelCategory(string fileName)
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

        private void ScanOnnxModels()
        {
            try
            {
                string root = AppDomain.CurrentDomain.BaseDirectory;
                var files = Directory.GetFiles(root, "*.onnx")
                    .Select(Path.GetFileName)
                    .Where(f => !string.IsNullOrEmpty(f))
                    .Select(f => f!)
                    .Where(f => !_config.ExcludedModels.Contains(f))
                    .OrderBy(f => f)
                    .ToList();

                if (files.Count == 0)
                {
                    WriteLog("ERROR: No usable .onnx model files found (all excluded or missing).");
                    StatusText.Text = "Mode: Model Missing";
                    CategoryComboBox.ItemsSource = null;
                    ModelComboBox.ItemsSource = null;
                    return;
                }

                _modelsByCategory = files
                    .GroupBy(GetModelCategory)
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
            InitializeAi(modelFileName);

            // モデル切替後、表示中ページをAI再変換
            if (_imageList.Count > 0)
            {
                RefreshDisplay();
            }
        }

        /// <summary>
        /// 現在選択中のモデルを「除外」リストへ登録し、選択肢から外す。
        /// 実用に耐えないモデル（極端に重い等）を今後選ばせないための恒久設定。
        /// </summary>
        private async void ExcludeModelButton_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            if (ModelComboBox.SelectedItem is not string modelFileName) return;

            var result = System.Windows.MessageBox.Show(
                $"「{modelFileName}」を選択肢から除外します。\n（設定ファイルに保存され、再起動後も除外されたままになります）\n\nよろしいですか？",
                "モデルの除外", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (result != MessageBoxResult.Yes) return;

            if (!_config.ExcludedModels.Contains(modelFileName))
            {
                _config.ExcludedModels.Add(modelFileName);
            }
            WriteLog($"Model excluded by user: {modelFileName}");

            ScanOnnxModels();
            SaveConfig();
            if (ModelComboBox.SelectedItem is string newModel)
            {
                await ApplyModelSelectionAsync(newModel);
            }
        }

        private void ApplyConfigToUi()
        {
            CheckSpread.IsChecked = _config.CheckSpread;
            CheckAutoDetect.IsChecked = _config.CheckAutoDetect;
            CheckPrefetch.IsChecked = _config.CheckPrefetch;
            SplitSlider.Value = _config.SplitSliderValue;

            CheckLens.IsChecked = _config.EnableLensCorrection;
            LensSlider.Value = _config.LensCorrectionAmount;

            CheckReticle.IsChecked = _config.ShowReticle;
            ReticleOverlay.Visibility = _config.ShowReticle ? Visibility.Visible : Visibility.Collapsed;
            if (LensShader != null)
            {
                LensShader.DistortionAmount = _config.EnableLensCorrection ? _config.LensCorrectionAmount : 0.0;
            }

            try
            {
                var obj = new BrushConverter().ConvertFromString(_config.BackgroundColor);
                if (obj is SolidColorBrush brush)
                {
                    RootGrid.Background = brush;
                }
            }
            catch (Exception ex)
            {
                WriteLog($"Background Apply Error: {ex.Message}");
            }
        }

        private void SetupEvents()
        {
            this.MouseRightButtonUp += (s, e) => ShowContextMenu();

            this.Drop += (s, e) =>
            {
                string[]? files = e.Data.GetData(DataFormats.FileDrop) as string[];
                if (files?.Length > 0) LoadSource(files[0]);
            };

            PageSlider.ValueChanged += (s, e) =>
            {
                if (_isUpdatingPageSliderInternal) return;
                RefreshDisplay();
            };

            SplitSlider.ValueChanged += (s, e) => UpdateImageDisplay();

            this.MouseMove += (s, e) =>
            {
                var p = e.GetPosition(this);
                AnimateSidebar(p.X > this.ActualWidth - (_isSidebarOpen ? 340 : 50) || Sidebar.IsMouseOver);
                AnimateBottomBar(p.Y > this.ActualHeight - 90 || BottomBar.IsMouseOver);
            };
        }

        public void LoadSource(string path)
        {
            WriteLog($"Loading source tree: {path}");
            _currentSourcePath = path;
            _imageList.Clear();
            int initialIndex = 0;
            try
            {
                if (Directory.Exists(path))
                {
                    var files = Directory.GetFiles(path)
                        .Where(f => !string.IsNullOrEmpty(f) && IsImageFile(f))
                        .OrderBy(f => f);
                    _imageList.AddRange(files);
                }
                else if (File.Exists(path))
                {
                    // パス自体が画像ファイルの場合は、親フォルダ内の画像を全てリスト化し、
                    // 自身の位置を初期表示ページとすることで、フォルダ指定時と同様に前後移動を可能にする
                    if (IsImageFile(path))
                    {
                        string? parentDir = Path.GetDirectoryName(path);
                        if (!string.IsNullOrEmpty(parentDir) && Directory.Exists(parentDir))
                        {
                            var files = Directory.GetFiles(parentDir)
                                .Where(f => !string.IsNullOrEmpty(f) && IsImageFile(f))
                                .OrderBy(f => f);
                            _imageList.AddRange(files);

                            int idx = _imageList.FindIndex(f => string.Equals(f, path, StringComparison.OrdinalIgnoreCase));
                            initialIndex = idx >= 0 ? idx : 0;
                        }
                        else
                        {
                            _imageList.Add(path);
                        }
                    }
                    else
                    {
                        string ext = Path.GetExtension(path).ToLower(CultureInfo.InvariantCulture);
                        if (ext == ".zip")
                        {
                            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                            using (ZipArchive archive = new ZipArchive(fs, ZipArchiveMode.Read))
                            {
                                var keys = archive.Entries
                                    .Where(e => !string.IsNullOrEmpty(e.FullName) && IsImageFile(e.FullName))
                                    .Select(e => e.FullName)
                                    .OrderBy(k => k);
                                _imageList.AddRange(keys);
                            }
                        }
                        else if (ext == ".rar" || ext == ".7z")
                        {
                            _imageList.AddRange(GetArchiveFileListViaCli(path));
                        }
                    }
                }
            }
            catch (Exception ex) { WriteLog($"Source Parser Error: {ex.Message}"); }

            if (_imageList.Count > 0)
            {
                if (initialIndex < 0 || initialIndex >= _imageList.Count) initialIndex = 0;

                _isUpdatingPageSliderInternal = true;
                PageSlider.Maximum = _imageList.Count - 1;
                PageSlider.Value = initialIndex;
                _isUpdatingPageSliderInternal = false;

                ResetTransform();
                DisplayPage(initialIndex);
            }
        }

        public void LoadImage(string path)
        {
            LoadSource(path);
        }

        private bool IsImageFile(string? f) =>
            !string.IsNullOrEmpty(f) && (
            f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase));

        private void RefreshDisplay() => DisplayPage((int)PageSlider.Value);

        private async void DisplayPage(int index)
        {
            _cts?.Cancel();
            var priorTask = _currentInferenceTask;
            if (priorTask != null)
            {
                // 前ページの推論が_currentCombinedOriginal等を参照中の可能性があるため、
                // 完全に終わってから破棄する（ObjectDisposedException対策）
                try { await priorTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }

            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            if (index < 0 || index >= _imageList.Count) return;

            try
            {
                _currentCombinedOriginal?.Dispose();
                _currentCombinedUpscaled?.Dispose();
                _currentCombinedUpscaled = null;

                Mat pRight = LoadMat(_imageList[index]);
                bool isAutoSingle = (CheckAutoDetect.IsChecked == true && (double)pRight.Width / pRight.Height > 1.1);
                bool isSpread = (CheckSpread.IsChecked == true && !isAutoSingle);

                Mat? pLeft = (isSpread && index + 1 < _imageList.Count) ? LoadMat(_imageList[index + 1]) : null;
                _currentCombinedOriginal = CombineMats(pRight, pLeft);

                PageText.Text = isSpread ? $"P.{index + 2}-{index + 1} / {_imageList.Count}" : $"P.{index + 1} / {_imageList.Count}";

                if ((int)PageSlider.Value != index)
                {
                    _isUpdatingPageSliderInternal = true;
                    PageSlider.Value = index;
                    _isUpdatingPageSliderInternal = false;
                }

                UpdateImageDisplay();

                if (_onnxSession == null && !string.IsNullOrEmpty(_config.SelectedModel))
                {
                    WriteLog("Session missing on display. Attempting automatic reinit.");
                    InitializeAi(_config.SelectedModel);
                }

                if (_onnxSession != null && _inputName != null)
                {
                    StatusText.Text = $"Processing... ({_activeEngineMode})";
                    var swTotal = System.Diagnostics.Stopwatch.StartNew();
                    try
                    {
                        var inferenceTask = Task.Run(() => PerformAiTiled(_currentCombinedOriginal, token), token);
                        _currentInferenceTask = inferenceTask;

                        // 30秒ごとに「まだ動いているか」をログへ出し、真のハングと単なる低速処理を切り分けやすくする
                        while (true)
                        {
                            var completed = await Task.WhenAny(inferenceTask, Task.Delay(30000));
                            if (completed == inferenceTask || token.IsCancellationRequested) break;
                            WriteLog($"Still processing... elapsed {swTotal.Elapsed.TotalSeconds:F0}s (model: {_config.SelectedModel})");
                        }

                        _currentCombinedUpscaled = await inferenceTask;
                        if (!token.IsCancellationRequested)
                        {
                            WriteLog($"AI processing completed in {swTotal.Elapsed.TotalSeconds:F1}s.");
                            StatusText.Text = $"Mode: {_activeEngineMode}";
                            UpdateImageDisplay();
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        WriteLog($"AI processing cancelled after {swTotal.Elapsed.TotalSeconds:F1}s.");
                    }
                    catch (Exception ex)
                    {
                        WriteLog($"Runtime Kernel Error after {swTotal.Elapsed.TotalSeconds:F1}s (model: {_config.SelectedModel}): {ex}");
                        StatusText.Text = "Mode: Error (see session.log)";

                        string msg = ex.ToString();
                        if (msg.Contains("CUBLAS", StringComparison.OrdinalIgnoreCase) ||
                            msg.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
                            msg.Contains("CUDNN", StringComparison.OrdinalIgnoreCase))
                        {
                            // GPUコンテキストが不安定化している可能性が高いため、
                            // 汚染されたセッションを使い回さず次回アクセス時に再構築させる
                            WriteLog("GPU runtime error detected. Disposing session to force clean reinit on next use.");
                            _onnxSession?.Dispose();
                            _onnxSession = null;
                            _inputName = null;
                        }
                    }
                    finally { _currentInferenceTask = null; }
                }
            }
            catch (Exception ex) { WriteLog($"Display Abend: {ex.Message}"); }
        }

        private Mat LoadMat(string key)
        {
            if (Directory.Exists(_currentSourcePath)) return Cv2.ImRead(key);

            if (File.Exists(key) && IsImageFile(key))
            {
                return Cv2.ImRead(key);
            }

            string ext = Path.GetExtension(_currentSourcePath ?? "").ToLower(CultureInfo.InvariantCulture);
            if (ext == ".zip")
            {
                using (FileStream fs = new FileStream(_currentSourcePath!, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (ZipArchive archive = new ZipArchive(fs, ZipArchiveMode.Read))
                {
                    ZipArchiveEntry? entry = archive.GetEntry(key);
                    if (entry != null)
                    {
                        using (Stream entryStream = entry.Open())
                        using (MemoryStream ms = new MemoryStream())
                        {
                            entryStream.CopyTo(ms);
                            return Cv2.ImDecode(ms.ToArray(), ImreadModes.Color);
                        }
                    }
                }
            }
            else if (ext == ".rar" || ext == ".7z")
            {
                return LoadViaSystemCli(_currentSourcePath!, key);
            }

            return new Mat();
        }

        private List<string> GetArchiveFileListViaCli(string archivePath)
        {
            var files = new List<string>();
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "7z.exe",
                Arguments = $"l \"{archivePath}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                using (var proc = System.Diagnostics.Process.Start(startInfo))
                {
                    if (proc != null)
                    {
                        using (var reader = proc.StandardOutput)
                        {
                            string? line;
                            bool isDataSection = false;
                            while ((line = reader.ReadLine()) != null)
                            {
                                if (line.Contains("-------------------"))
                                {
                                    isDataSection = !isDataSection;
                                    continue;
                                }
                                if (isDataSection && line.Length > 53)
                                {
                                    string filename = line.Substring(53).Trim();
                                    if (IsImageFile(filename)) files.Add(filename);
                                }
                            }
                        }
                        proc.WaitForExit();
                    }
                }
            }
            catch (Exception ex) { WriteLog($"CLI List Error: {ex.Message}"); }
            return files.OrderBy(f => f).ToList();
        }

        private Mat LoadViaSystemCli(string archivePath, string entryKey)
        {
            if (!Directory.Exists(_tempExtractDir)) Directory.CreateDirectory(_tempExtractDir);

            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "7z.exe",
                Arguments = $"e \"{archivePath}\" -o\"{_tempExtractDir}\" \"{entryKey}\" -y",
                WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden,
                CreateNoWindow = true,
                UseShellExecute = false
            };

            try
            {
                using (var proc = System.Diagnostics.Process.Start(startInfo))
                {
                    proc?.WaitForExit();
                }

                string extractedPath = Path.Combine(_tempExtractDir, Path.GetFileName(entryKey));
                if (File.Exists(extractedPath))
                {
                    Mat mat = Cv2.ImRead(extractedPath);
                    try { File.Delete(extractedPath); } catch { }
                    return mat;
                }
            }
            catch (Exception ex) { WriteLog($"CLI Extracted Image Error: {ex.Message}"); }
            return new Mat();
        }

        private Mat CombineMats(Mat r, Mat? l)
        {
            if (l == null) return r.Clone();
            Mat res = new Mat(Math.Max(r.Height, l.Height), r.Width + l.Width, r.Type(), Scalar.Black);
            using (var roiL = new Mat(res, new OpenCvSharp.Rect(0, 0, l.Width, l.Height))) l.CopyTo(roiL);
            using (var roiR = new Mat(res, new OpenCvSharp.Rect(l.Width, 0, r.Width, r.Height))) r.CopyTo(roiR);
            r.Dispose(); l.Dispose();
            return res;
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

        private Mat PerformAiTiled(Mat input, CancellationToken token)
        {
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
            WriteLog($"[AI] Start: {inWidth}x{inHeight}, model={_config.SelectedModel}, tileSize={tileSize}, " +
                     $"fixedInput={(_fixedInputSize?.ToString() ?? "dynamic")}, tiles={tileCountX * tileCountY} ({tileCountX}x{tileCountY})");

            // 最初のタイルを実際に推論し、出力テンソルの実寸からスケール倍率を算出する
            // （1x/2x/3x/4x等、モデルごとに異なるため決め打ちにしない）
            var swProbe = System.Diagnostics.Stopwatch.StartNew();
            int probeCw = Math.Min(tileSize + overlap, inWidth);
            int probeCh = Math.Min(tileSize + overlap, inHeight);
            int probeTargetW = GetPaddedTargetSize(probeCw);
            int probeTargetH = GetPaddedTargetSize(probeCh);

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

            int tileIndex = 0;
            var swTile = System.Diagnostics.Stopwatch.StartNew();

            for (int y = 0; y < inHeight; y += tileSize)
            {
                for (int x = 0; x < inWidth; x += tileSize)
                {
                    token.ThrowIfCancellationRequested();
                    tileIndex++;
                    if (x == 0 && y == 0) continue; // プローブで処理済み

                    int cw = Math.Min(tileSize + overlap, inWidth - x);
                    int ch = Math.Min(tileSize + overlap, inHeight - y);
                    int targetW = GetPaddedTargetSize(cw);
                    int targetH = GetPaddedTargetSize(ch);

                    swTile.Restart();
                    Mat up;
                    using (var tile = new Mat(input, new OpenCvSharp.Rect(x, y, cw, ch)))
                    using (var padded = PadToTarget(tile, targetW, targetH))
                    {
                        up = ProcessTile(padded);
                    }
                    WriteLog($"[AI] Tile {tileIndex}/{tileCountX * tileCountY} (x={x},y={y},{cw}x{ch} padded->{targetW}x{targetH}) " +
                             $"-> {up.Width}x{up.Height} in {swTile.Elapsed.TotalMilliseconds:F0}ms");

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
            WriteLog($"[AI] All {tileCountX * tileCountY} tiles done. Output: {output.Width}x{output.Height}");
            return output;
        }

        [ThreadStatic] private static float[]? _tileInputBuffer;

        private Mat ProcessTile(Mat tile)
        {
            int w = tile.Width, h = tile.Height;
            using var rgb = new Mat();
            Cv2.CvtColor(tile, rgb, ColorConversionCodes.BGR2RGB);

            int required = 3 * w * h;
            if (_tileInputBuffer == null || _tileInputBuffer.Length < required)
            {
                _tileInputBuffer = new float[required];
            }
            float[] data = _tileInputBuffer;

            var idx = rgb.GetUnsafeGenericIndexer<Vec3b>();
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    var v = idx[y, x];
                    data[0 * w * h + y * w + x] = v.Item0 / 255f;
                    data[1 * w * h + y * w + x] = v.Item1 / 255f;
                    data[2 * w * h + y * w + x] = v.Item2 / 255f;
                }
            }

            // バッファを使い回すため、テンソルには実サイズ分だけを渡す（末尾の余剰領域は無視される）
            var tensor = required == data.Length
                ? new DenseTensor<float>(data, new[] { 1, 3, h, w })
                : new DenseTensor<float>(new Memory<float>(data, 0, required), new[] { 1, 3, h, w });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName ?? "input", tensor) };
            using var results = _onnxSession!.Run(inputs);
            var output = results.First().AsTensor<float>();

            // 出力サイズをテンソルの実寸から取得する（モデルのスケール倍率を仮定しない）
            int outH = output.Dimensions[2];
            int outW = output.Dimensions[3];

            Mat res = new Mat(outH, outW, MatType.CV_8UC3);
            var resIdx = res.GetUnsafeGenericIndexer<Vec3b>();
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    resIdx[y, x] = new Vec3b(
                        (byte)Math.Clamp(output[0, 2, y, x] * 255, 0, 255),
                        (byte)Math.Clamp(output[0, 1, y, x] * 255, 0, 255),
                        (byte)Math.Clamp(output[0, 0, y, x] * 255, 0, 255)
                    );
                }
            }
            return res;
        }

        private void UpdateImageDisplay()
        {
            if (_currentCombinedOriginal == null) return;
            if (_currentCombinedUpscaled == null)
            {
                MainImage.Source = _currentCombinedOriginal.ToWriteableBitmap();
                return;
            }
            if (SplitSlider.Value >= 100)
            {
                MainImage.Source = _currentCombinedUpscaled.ToWriteableBitmap();
                return;
            }

            int w = _currentCombinedUpscaled.Width, h = _currentCombinedUpscaled.Height;
            int sx = (int)(w * (SplitSlider.Value / 100.0));
            Mat disp = new Mat(h, w, _currentCombinedUpscaled.Type());

            if (sx > 0)
            {
                using var roiAi = new Mat(disp, new OpenCvSharp.Rect(0, 0, sx, h));
                _currentCombinedUpscaled.SubMat(0, h, 0, sx).CopyTo(roiAi);
            }
            if (sx < w)
            {
                using var tmp = new Mat();
                Cv2.Resize(_currentCombinedOriginal, tmp, new OpenCvSharp.Size(w, h));
                using var roiRaw = new Mat(disp, new OpenCvSharp.Rect(sx, 0, w - sx, h));
                tmp.SubMat(0, h, sx, w).CopyTo(roiRaw);
            }
            if (sx > 25 && sx < w - 25)
            {
                Cv2.Rectangle(disp, new OpenCvSharp.Rect(sx - 25, 0, 50, h), new Scalar(0, 0, 255), -1);
            }

            MainImage.Source = disp.ToWriteableBitmap();
            disp.Dispose();
        }

        private void MovePage(int dir)
        {
            int step = (CheckSpread.IsChecked == true) ? 2 : 1;
            int next = (int)PageSlider.Value + (dir * step);
            if (next < 0) LoadNextArchive(-1);
            else if (next > PageSlider.Maximum) LoadNextArchive(1);
            else DisplayPage(next);
        }

        private void LoadNextArchive(int dir)
        {
            if (string.IsNullOrEmpty(_currentSourcePath)) return;
            var parent = Path.GetDirectoryName(_currentSourcePath);
            if (parent == null) return;
            var archives = Directory.GetFiles(parent, "*.*")
                .Where(f => f.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".rar", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".7z", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f).ToList();

            int idx = archives.IndexOf(_currentSourcePath) + dir;
            if (idx >= 0 && idx < archives.Count)
            {
                LoadSource(archives[idx]);
                if (dir < 0) DisplayPage((int)PageSlider.Maximum);
            }
        }

        private void ResetTransform()
        {
            ImgScale.ScaleX = 1.0;
            ImgScale.ScaleY = 1.0;
            ImgTranslate.X = 0;
            ImgTranslate.Y = 0;
        }

        #region マウスインタラクションコントロール

        private void ImageContainer_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (MainImage.Source == null) return;
            _isDragging = true;
            _startPoint = e.GetPosition(RootGrid);
            _origin = new System.Windows.Point(ImgTranslate.X, ImgTranslate.Y);
            ImageContainer.CaptureMouse();
        }

        private void ImageContainer_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            _isDragging = false;
            ImageContainer.ReleaseMouseCapture();
        }

        private void ImageContainer_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_isDragging) return;
            System.Windows.Point currentPoint = e.GetPosition(RootGrid);
            ImgTranslate.X = _origin.X + (currentPoint.X - _startPoint.X);
            ImgTranslate.Y = _origin.Y + (currentPoint.Y - _startPoint.Y);
        }

        private void ImageContainer_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (MainImage.Source == null) return;

            if (Keyboard.Modifiers != ModifierKeys.Shift)
            {
                MovePage(e.Delta > 0 ? -1 : 1);
            }
            else
            {
                System.Windows.Point mousePos = e.GetPosition(ImageBorder);
                double zoomFactor = e.Delta > 0 ? 1.15 : (1.0 / 1.15);

                double newScaleX = ImgScale.ScaleX * zoomFactor;
                if (newScaleX < 0.1 || newScaleX > 20.0) return;

                ImgScale.ScaleX = newScaleX;
                ImgScale.ScaleY = newScaleX;

                ImgTranslate.X -= (mousePos.X - ImageBorder.ActualWidth / 2) * (zoomFactor - 1) * ImgScale.ScaleX;
                ImgTranslate.Y -= (mousePos.Y - ImageBorder.ActualHeight / 2) * (zoomFactor - 1) * ImgScale.ScaleY;
            }
        }
        #endregion

        #region カラーピッカー関連のHSV演算・プレビュー制御

        private void OnSvCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            UpdateHsvFromMouse(e.GetPosition(ColorPickerSvCanvas));
        }

        private void OnSvCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                UpdateHsvFromMouse(e.GetPosition(ColorPickerSvCanvas));
            }
        }

        private void OnHueSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            _currentHue = SliderHue.Value;
            UpdateColorPickerBrush();
            UpdatePreview();
        }

        private void UpdateHsvFromMouse(System.Windows.Point p)
        {
            double x = Math.Clamp(p.X, 0, ColorPickerSvCanvas.ActualWidth);
            double y = Math.Clamp(p.Y, 0, ColorPickerSvCanvas.ActualHeight);

            _currentSaturation = x / ColorPickerSvCanvas.ActualWidth;
            _currentValue = 1.0 - (y / ColorPickerSvCanvas.ActualHeight);

            UpdatePreview();
        }

        private void UpdateColorPickerBrush()
        {
            // SV Canvasの背景に Hue に対応する横方向グラデーション（黒～純色～白）を設定
            var gradientBrush = new LinearGradientBrush
            {
                StartPoint = new System.Windows.Point(0, 0),
                EndPoint = new System.Windows.Point(1, 0)
            };

            System.Windows.Media.Color pureColor = HsvToRgb(_currentHue, 1.0, 1.0);
            gradientBrush.GradientStops.Add(new GradientStop(Colors.White, 0.0));
            gradientBrush.GradientStops.Add(new GradientStop(pureColor, 1.0));

            var verticalBrush = new DrawingBrush
            {
                Stretch = Stretch.Fill,
                Drawing = new GeometryDrawing
                {
                    Geometry = new RectangleGeometry(new System.Windows.Rect(0, 0, 1, 1)),
                    Brush = gradientBrush
                }
            };

            // 上下が黒から透明のレイヤーを重ねてグラデーションを構築
            var blackToTransparent = new LinearGradientBrush
            {
                StartPoint = new System.Windows.Point(0, 1),
                EndPoint = new System.Windows.Point(0, 0)
            };
            blackToTransparent.GradientStops.Add(new GradientStop(Colors.Black, 0.0));
            blackToTransparent.GradientStops.Add(new GradientStop(Colors.Transparent, 1.0));

            var group = new DrawingGroup();
            group.Children.Add(new ImageDrawing(null, new System.Windows.Rect(0, 0, 1, 1))); // ベース
            group.Children.Add(new GeometryDrawing(gradientBrush, null, new RectangleGeometry(new System.Windows.Rect(0, 0, 1, 1))));
            group.Children.Add(new GeometryDrawing(blackToTransparent, null, new RectangleGeometry(new System.Windows.Rect(0, 0, 1, 1))));

            ColorPickerSvCanvas.Background = new DrawingBrush(group);
        }

        private void UpdatePreview()
        {
            System.Windows.Media.Color rgb = HsvToRgb(_currentHue, _currentSaturation, _currentValue);
            _selectedHexColor = $"#{rgb.R:X2}{rgb.G:X2}{rgb.B:X2}";

            PreviewColorBox.Fill = new SolidColorBrush(rgb);
            HsvValueText.Text = $"H: {(int)_currentHue}°, S: {(int)(_currentSaturation * 100)}%, V: {(int)(_currentValue * 100)}%";
        }

        private System.Windows.Media.Color HsvToRgb(double h, double s, double v)
        {
            double c = v * s;
            double x = c * (1.0 - Math.Abs((h / 60.0) % 2.0 - 1.0));
            double m = v - c;

            double r = 0, g = 0, b = 0;

            if (h >= 0 && h < 60) { r = c; g = x; b = 0; }
            else if (h >= 60 && h < 120) { r = x; g = c; b = 0; }
            else if (h >= 120 && h < 180) { r = 0; g = c; b = x; }
            else if (h >= 180 && h < 240) { r = 0; g = x; b = c; }
            else if (h >= 240 && h < 300) { r = x; g = 0; b = c; }
            else if (h >= 300 && h < 360) { r = c; g = 0; b = x; }

            return System.Windows.Media.Color.FromRgb(
                (byte)Math.Clamp((r + m) * 255, 0, 255),
                (byte)Math.Clamp((g + m) * 255, 0, 255),
                (byte)Math.Clamp((b + m) * 255, 0, 255)
            );
        }

        private void OnApplyCustomColorClick(object sender, RoutedEventArgs e)
        {
            try
            {
                var brush = new SolidColorBrush(HsvToRgb(_currentHue, _currentSaturation, _currentValue));
                RootGrid.Background = brush;
                SaveConfig();
                ShowNotification($"背景色を変更: {_selectedHexColor}");
            }
            catch (Exception ex)
            {
                WriteLog($"Custom Color Apply Error: {ex.Message}");
            }
        }

        #endregion

        #region 各種イベント・ショートカット・アシスト制御

        protected override void OnPreviewKeyDown(KeyEventArgs e)
        {
            base.OnPreviewKeyDown(e);

            switch (e.Key)
            {
                case Key.Space:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        int next = (int)PageSlider.Value + 1;
                        if (next > PageSlider.Maximum) LoadNextArchive(1);
                        else DisplayPage(next);
                    }
                    else
                    {
                        MovePage(1);
                    }
                    e.Handled = true;
                    break;
                case Key.Left:
                case Key.OemComma:
                    MovePage(-1);
                    e.Handled = true;
                    break;
                case Key.Right:
                case Key.OemPeriod:
                    MovePage(1);
                    e.Handled = true;
                    break;
                case Key.Enter:
                    ToggleFullscreen();
                    e.Handled = true;
                    break;
                case Key.Up:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        ZoomAtCursor(0.1);
                        e.Handled = true;
                    }
                    break;
                case Key.Down:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        ZoomAtCursor(-0.1);
                        e.Handled = true;
                    }
                    break;
                case Key.R:
                    _config.ShowReticle = !_config.ShowReticle;
                    ApplyConfigToUi();
                    ShowNotification(_config.ShowReticle ? "レティクル: ON" : "レティクル: OFF");
                    e.Handled = true;
                    break;
                case Key.K:
                    _config.EnableLensCorrection = !_config.EnableLensCorrection;
                    ApplyConfigToUi();
                    ShowNotification(_config.EnableLensCorrection ? $"逆湾曲補正: ON (強度: {_config.LensCorrectionAmount:F2})" : "逆湾曲補正: OFF");
                    e.Handled = true;
                    break;
                case Key.OemOpenBrackets:
                    if (_config.EnableLensCorrection)
                    {
                        _config.LensCorrectionAmount = Math.Round(_config.LensCorrectionAmount - 0.01, 2);
                        ApplyConfigToUi();
                        ShowNotification($"補正強度: {_config.LensCorrectionAmount:F2}");
                    }
                    e.Handled = true;
                    break;

                case Key.OemCloseBrackets:
                    if (_config.EnableLensCorrection)
                    {
                        _config.LensCorrectionAmount = Math.Round(_config.LensCorrectionAmount + 0.01, 2);
                        ApplyConfigToUi();
                        ShowNotification($"補正強度: {_config.LensCorrectionAmount:F2}");
                    }
                    e.Handled = true;
                    break;
                case Key.Escape:
                    Application.Current.Shutdown();
                    break;
            }
        }

        private void OnSettingChanged(object sender, RoutedEventArgs e) => RefreshDisplay();

        private void OnLensSettingChanged(object sender, RoutedEventArgs e)
        {
            _config.EnableLensCorrection = CheckLens.IsChecked ?? false;
            ApplyConfigToUi();
            SaveConfig();
        }

        private void OnReticleSettingChanged(object sender, RoutedEventArgs e)
        {
            if (CheckReticle.IsChecked == true)
            {
                ReticleOverlay.Visibility = Visibility.Visible;
            }
            else
            {
                ReticleOverlay.Visibility = Visibility.Collapsed;
            }
        }

        private void OnLensSliderChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (LensShader == null) return;
            _config.LensCorrectionAmount = LensSlider.Value;
            if (_config.EnableLensCorrection)
            {
                LensShader.DistortionAmount = _config.LensCorrectionAmount;
            }
        }

        private void AnimateSidebar(bool show)
        {
            if (_isSidebarOpen == show) return;
            _isSidebarOpen = show;
            SidebarTransform.BeginAnimation(TranslateTransform.XProperty, new DoubleAnimation(show ? 0 : 320, TimeSpan.FromMilliseconds(200)));
        }

        private void AnimateBottomBar(bool show)
        {
            if (_isBottomBarOpen == show) return;
            _isBottomBarOpen = show;
            BottomBarTransform.BeginAnimation(TranslateTransform.YProperty, new DoubleAnimation(show ? 0 : 110, TimeSpan.FromMilliseconds(200)));
        }

        private void MenuButton_Click(object sender, RoutedEventArgs e)
        {
            var menu = new System.Windows.Controls.ContextMenu();
            var itemFullscreen = new System.Windows.Controls.MenuItem { Header = "全画面表示の切り替え" };
            itemFullscreen.Click += (s, ev) => ToggleFullscreen();
            menu.Items.Add(itemFullscreen);

            menu.Items.Add(new System.Windows.Controls.Separator());

            var itemExit = new System.Windows.Controls.MenuItem { Header = "アプリケーションを終了" };
            itemExit.Click += (s, ev) => Application.Current.Shutdown();
            menu.Items.Add(itemExit);

            menu.PlacementTarget = (System.Windows.Controls.Button)sender;
            menu.Placement = System.Windows.Controls.Primitives.PlacementMode.Bottom;
            menu.IsOpen = true;
        }

        private void ZoomAtCursor(double delta)
        {
            System.Windows.Point mousePos = Mouse.GetPosition(ImageContainer);
            double scale = ImgScale.ScaleX;
            double newScale = Math.Max(0.1, Math.Min(5.0, scale + delta));
            double ratio = newScale / scale;

            ImgTranslate.X = (ImgTranslate.X - mousePos.X) * ratio + mousePos.X;
            ImgTranslate.Y = (ImgTranslate.Y - mousePos.Y) * ratio + mousePos.Y;

            ImgScale.ScaleX = newScale;
            ImgScale.ScaleY = newScale;
        }

        private void ShowContextMenu()
        {
            var menu = new System.Windows.Controls.ContextMenu();

            var itemFullscreen = new System.Windows.Controls.MenuItem { Header = "全画面表示の切り替え" };
            itemFullscreen.Click += (s, ev) => ToggleFullscreen();
            menu.Items.Add(itemFullscreen);

            menu.Items.Add(new System.Windows.Controls.Separator());

            var itemLensToggle = new System.Windows.Controls.MenuItem { Header = "レンズ補正 (K) - ON/OFF" };
            itemLensToggle.Click += (s, ev) => {
                _config.EnableLensCorrection = !_config.EnableLensCorrection;
                ApplyConfigToUi();
            };
            menu.Items.Add(itemLensToggle);

            var itemLensInc = new System.Windows.Controls.MenuItem { Header = "補正値アップ (]) +0.01" };
            itemLensInc.Click += (s, ev) => AdjustLensCorrection(0.01);
            menu.Items.Add(itemLensInc);

            var itemLensDec = new System.Windows.Controls.MenuItem { Header = "補正値ダウン ([) -0.01" };
            itemLensDec.Click += (s, ev) => AdjustLensCorrection(-0.01);
            menu.Items.Add(itemLensDec);

            menu.Items.Add(new System.Windows.Controls.Separator());

            var itemExit = new System.Windows.Controls.MenuItem { Header = "アプリケーションを終了" };
            itemExit.Click += (s, ev) => Application.Current.Shutdown();
            menu.Items.Add(itemExit);

            menu.IsOpen = true;
        }

        private void AdjustLensCorrection(double delta)
        {
            if (!_config.EnableLensCorrection) _config.EnableLensCorrection = true;
            _config.LensCorrectionAmount = Math.Round(_config.LensCorrectionAmount + delta, 2);
            ApplyConfigToUi();
            ShowNotification($"補正強度: {_config.LensCorrectionAmount:F2}");
        }

        private void ToggleFullscreen()
        {
            if (this.WindowStyle == WindowStyle.None)
            {
                this.WindowStyle = WindowStyle.SingleBorderWindow; this.WindowState = WindowState.Normal;
            }
            else
            {
                this.WindowStyle = WindowStyle.None; this.WindowState = WindowState.Maximized;
            }
        }

        private void UpdateMemoryUsage() => MemoryText.Text = $"RAM: {GC.GetTotalMemory(false) / 1024 / 1024} MB";

        private async void ShowNotification(string message)
        {
            NotificationText.Text = message;
            NotificationBadge.Visibility = Visibility.Visible;
            await Task.Delay(1800);
            if (NotificationText.Text == message)
            {
                NotificationBadge.Visibility = Visibility.Collapsed;
            }
        }

        #endregion
    }

    /// <summary>
    /// UIレイアウト中央座標導出用コンバーター
    /// </summary>
    public class HalfLayoutConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double doubleValue) return doubleValue / 2.0;
            return 0.0;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
    }
}