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
        public bool CheckAutoDetect { get; set; } = false;
        public bool CheckPrefetch { get; set; } = false;
        public double SplitSliderValue { get; set; } = 100;
        public string LastSourcePath { get; set; } = string.Empty;

        // レンズ補正およびレティクル用永続パラメータ
        public bool ShowReticle { get; set; } = true;
        public bool EnableLensCorrection { get; set; } = false;
        public double LensCorrectionAmount { get; set; } = 0.40;
    }

    public partial class MainWindow : System.Windows.Window
    {
        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);
        private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

        // AI・描画関連のメンバ
        private InferenceSession? _onnxSession;
        private string? _inputName;
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

        // 無限ループイベントを抑止するためのフラグ
        private bool _isUpdatingPageSliderInternal = false;

        public MainWindow(string[]? args = null) // 引数を受け取れるように変更)
        {
            InitializeComponent();
            InitLog();
            WriteLog("ZiView Engine Starting (Native Core Standard)...");

            LoadConfig();

            // コマンドライン引数が存在する場合のみ、初期パスとして採用
            if (args != null && args.Length > 0)
            {
                _currentSourcePath = args[0];
            }

            this.SourceInitialized += (s, e) =>
            {
                IntPtr hwnd = new WindowInteropHelper(this).Handle;
                int darkMode = 1;
                DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ref darkMode, sizeof(int));
            };

            _monitorTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
            _monitorTimer.Tick += (s, e) => UpdateMemoryUsage();
            _monitorTimer.Start();

            this.Loaded += (s, e) =>
            {
                InitializeAi();
                ApplyConfigToUi();
                // 設定ファイルまたは引数でセットされたパスがあれば一度だけ読み込む
                if (!string.IsNullOrEmpty(_currentSourcePath))
                {
                    LoadSource(_currentSourcePath);
                }
            };
            SetupEvents();
        }

        private void InitLog()
        {
            try { File.WriteAllText(_logPath, $"=== Session Started at {DateTime.Now} ===\n"); } catch { }
        }

        private void WriteLog(string message)
        {
            try { File.AppendAllText(_logPath, $"[{DateTime.Now:HH:mm:ss}] {message}\n"); } catch { }
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
                _config.CheckSpread = CheckSpread.IsChecked ?? false;
                _config.CheckAutoDetect = CheckAutoDetect.IsChecked ?? false;
                _config.CheckPrefetch = CheckPrefetch.IsChecked ?? true;
                _config.SplitSliderValue = SplitSlider.Value;
                _config.LastSourcePath = _currentSourcePath ?? string.Empty;

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

            // 一時展開フォルダのクリーニング
            try
            {
                if (Directory.Exists(_tempExtractDir))
                {
                    Directory.Delete(_tempExtractDir, true);
                }
            }
            catch { }
        }

        private void InitializeAi()
        {
            try
            {
                string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "RealESRGAN_x4plus_anime_6B.onnx");
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
                    WriteLog("Attempting TensorRT/CUDA...");
                    options.AppendExecutionProvider_Tensorrt(0);
                    options.AppendExecutionProvider_CUDA(0);
                    _activeEngineMode = "RTX (TensorRT)";
                }
                catch (Exception ex)
                {
                    WriteLog($"TensorRT/CUDA down: {ex.Message}");
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

                _onnxSession = new InferenceSession(modelPath, options);
                _inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault();
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

        private void ApplyConfigToUi()
        {
            CheckSpread.IsChecked = _config.CheckSpread;
            CheckAutoDetect.IsChecked = _config.CheckAutoDetect;
            CheckPrefetch.IsChecked = _config.CheckPrefetch;
            SplitSlider.Value = _config.SplitSliderValue;

            CheckLens.IsChecked = _config.EnableLensCorrection;
            LensSlider.Value = _config.LensCorrectionAmount;

            ReticleOverlay.Visibility = !_config.ShowReticle ? Visibility.Collapsed : Visibility.Visible;
            if (LensShader != null)
            {
                LensShader.DistortionAmount = _config.EnableLensCorrection ? _config.LensCorrectionAmount : 0.0;
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

            // ページスライダー変更時のイベント処理
            PageSlider.ValueChanged += (s, e) =>
            {
                if (_isUpdatingPageSliderInternal) return;
                RefreshDisplay();
            };

            SplitSlider.ValueChanged += (s, e) => UpdateImageDisplay();

            this.MouseMove += (s, e) =>
            {
                var p = e.GetPosition(this);
                AnimateSidebar(p.X > this.ActualWidth - (_isSidebarOpen ? 300 : 50));
                AnimateBottomBar(p.Y > this.ActualHeight - 60);
            };
        }

        public void LoadSource(string path)
        {
            WriteLog($"Loading source tree: {path}");
            _currentSourcePath = path;
            _imageList.Clear();
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
            catch (Exception ex) { WriteLog($"Source Parser Error: {ex.Message}"); }

            if (_imageList.Count > 0)
            {
                _isUpdatingPageSliderInternal = true;
                PageSlider.Maximum = _imageList.Count - 1;
                PageSlider.Value = 0;
                _isUpdatingPageSliderInternal = false;

                ResetTransform();
                RefreshDisplay();
            }

        }

        // App.xaml.cs から呼び出される互換用ブリッジメソッド
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

                // テキスト表示を正確に更新
                PageText.Text = isSpread ? $"P.{index + 2}-{index + 1} / {_imageList.Count}" : $"P.{index + 1} / {_imageList.Count}";

                if ((int)PageSlider.Value != index)
                {
                    _isUpdatingPageSliderInternal = true;
                    PageSlider.Value = index;
                    _isUpdatingPageSliderInternal = false;
                }

                UpdateImageDisplay();

                if (_onnxSession != null && _inputName != null)
                {
                    StatusText.Text = $"Processing... ({_activeEngineMode})";
                    try
                    {
                        _currentCombinedUpscaled = await Task.Run(() => PerformAiTiled(_currentCombinedOriginal, token), token);
                        if (!token.IsCancellationRequested)
                        {
                            StatusText.Text = $"Mode: {_activeEngineMode}";
                            UpdateImageDisplay();
                        }
                    }
                    catch (OperationCanceledException) { }
                    catch (Exception ex) { WriteLog($"Runtime Kernel Error: {ex.Message}"); }
                }
            }
            catch (Exception ex) { WriteLog($"Display Abend: {ex.Message}"); }
            finally { GC.Collect(); }
        }

        private Mat LoadMat(string key)
        {
            if (Directory.Exists(_currentSourcePath)) return Cv2.ImRead(key);

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

        private Mat PerformAiTiled(Mat input, CancellationToken token)
        {
            int tileSize = 256;
            int overlap = 16;
            int scale = 4;

            int inWidth = input.Width;
            int inHeight = input.Height;

            Mat output = new Mat(inHeight * scale, inWidth * scale, input.Type());

            for (int y = 0; y < inHeight; y += tileSize)
            {
                for (int x = 0; x < inWidth; x += tileSize)
                {
                    token.ThrowIfCancellationRequested();

                    int cw = Math.Min(tileSize + overlap, inWidth - x);
                    int ch = Math.Min(tileSize + overlap, inHeight - y);

                    using var tile = new Mat(input, new OpenCvSharp.Rect(x, y, cw, ch));
                    using var up = ProcessTile(tile);

                    int ow = Math.Min(tileSize * scale, output.Width - x * scale);
                    int oh = Math.Min(tileSize * scale, output.Height - y * scale);
                    using var roiSrc = new Mat(up, new OpenCvSharp.Rect(0, 0, ow, oh));
                    using var roiDst = new Mat(output, new OpenCvSharp.Rect(x * scale, y * scale, ow, oh));
                    roiSrc.CopyTo(roiDst);
                }
            }
            return output;
        }

        private Mat ProcessTile(Mat tile)
        {
            int w = tile.Width, h = tile.Height;
            using var rgb = new Mat();
            Cv2.CvtColor(tile, rgb, ColorConversionCodes.BGR2RGB);

            float[] data = new float[3 * w * h];
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

            var tensor = new DenseTensor<float>(data, new[] { 1, 3, h, w });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName ?? "input", tensor) };
            using var results = _onnxSession!.Run(inputs);
            var output = results.First().AsTensor<float>();

            Mat res = new Mat(h * 4, w * 4, MatType.CV_8UC3);
            var resIdx = res.GetUnsafeGenericIndexer<Vec3b>();
            for (int y = 0; y < h * 4; y++)
            {
                for (int x = 0; x < w * 4; x++)
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
            if (_currentCombinedUpscaled == null || SplitSlider.Value >= 100)
            {
                MainImage.Source = _currentCombinedOriginal.ToWriteableBitmap();
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

            // Shiftキーなしでページめくり（最優先操作）
            if (Keyboard.Modifiers != ModifierKeys.Shift)
            {
                MovePage(e.Delta > 0 ? -1 : 1);
            }
            // Shiftキー押下時は拡大縮小
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

        #region 各種イベント・ショートカット・アシスト制御

        protected override void OnPreviewKeyDown(KeyEventArgs e)
        {
            base.OnPreviewKeyDown(e);

            switch (e.Key)
            {
                case Key.Space:
                    if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                    {
                        // Shift+Space: 見開き設定に関わらず、強制的に1ページずつ進める
                        int next = (int)PageSlider.Value + 1;
                        if (next > PageSlider.Maximum) LoadNextArchive(1);
                        else DisplayPage(next);
                    }
                    else
                    {
                        // 通常Space: MovePageメソッド経由（見開きなら2ページ飛び）
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
                        // 0.01単位へ変更
                        _config.LensCorrectionAmount = Math.Round(_config.LensCorrectionAmount - 0.01, 2);
                        ApplyConfigToUi();
                        ShowNotification($"補正強度: {_config.LensCorrectionAmount:F2}");
                    }
                    e.Handled = true;
                    break;

                case Key.OemCloseBrackets:
                    if (_config.EnableLensCorrection)
                    {
                        // 0.01単位へ変更
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

        //「レンズ補正（ディストーション補正）を行う際の『光学中心（光軸）』を視覚的に合わせるため」のガイドライン表示切替
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
            SidebarTransform.BeginAnimation(TranslateTransform.XProperty, new DoubleAnimation(show ? 0 : 260, TimeSpan.FromMilliseconds(200)));
        }

        private void AnimateBottomBar(bool show)
        {
            if (_isBottomBarOpen == show) return;
            _isBottomBarOpen = show;
            BottomBarTransform.BeginAnimation(TranslateTransform.YProperty, new DoubleAnimation(show ? 0 : 80, TimeSpan.FromMilliseconds(200)));
        }

        // このコードは必要ないのですが、将来的な機能拡張のために残しておきます
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
        private void ShowContextMenu()
        {
            var menu = new System.Windows.Controls.ContextMenu();

            // 全画面表示切り替え
            var itemFullscreen = new System.Windows.Controls.MenuItem { Header = "全画面表示の切り替え" };
            itemFullscreen.Click += (s, ev) => ToggleFullscreen();
            menu.Items.Add(itemFullscreen);

            // --- ここから追加 ---
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
            // --- ここまで追加 ---

            menu.Items.Add(new System.Windows.Controls.Separator());

            // アプリ終了
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