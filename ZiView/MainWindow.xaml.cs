using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;

using OpenCvSharp;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: アプリケーションのコア部分。
    /// ウィンドウのライフサイクル、設定の読み書き、ログ基盤を担当する。
    /// AI推論(MainWindow.AiEngine.cs)、画像表示(MainWindow.Display.cs)、
    /// 入力操作(MainWindow.Input.cs)、カラーピッカー(MainWindow.ColorPicker.cs)は別ファイルに分割。
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);
        private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

        private readonly DispatcherTimer _monitorTimer;

        // パス定義
        private readonly string _configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "zi_view_config.json");
        private readonly string _logPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "session.log");

        private AppConfig _config = new();
        private CancellationTokenSource? _cts;
        private Task? _currentInferenceTask;

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
                _config.EnableAiInference = CheckAiEnable.IsChecked ?? true;
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

        private void ApplyConfigToUi()
        {
            CheckSpread.IsChecked = _config.CheckSpread;
            CheckAutoDetect.IsChecked = _config.CheckAutoDetect;
            CheckPrefetch.IsChecked = _config.CheckPrefetch;
            SplitSlider.Value = _config.SplitSliderValue;

            CheckAiEnable.IsChecked = _config.EnableAiInference;

            CheckLens.IsChecked = _config.EnableLensCorrection;
            LensSlider.Value = _config.LensCorrectionAmount;

            CheckReticle.IsChecked = _config.ShowReticle;
            ReticleOverlay.Visibility = _config.ShowReticle ? Visibility.Visible : Visibility.Collapsed;
            if (LensShader != null)
            {
                LensShader.DistortionAmount = _config.EnableLensCorrection ? _config.LensCorrectionAmount : 0.0;
            }

            PopulateOsdPositionComboBox();
            ApplyAiOsdPosition();

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
    }
}
