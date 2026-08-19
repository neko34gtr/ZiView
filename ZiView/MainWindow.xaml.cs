using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Channels;
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

        // ログ出力先: LoadConfig()完了後、コンストラクタ内でResolveLogPath()により確定する
        // （設定ウィンドウでのカスタムパス指定を反映するため、フィールド初期化子ではなく明示的に遅延させる）。
        private string _logPath = string.Empty;

        // ログ書き込みを非同期化するための一本化されたチャンネル。
        // 呼び出し側（AI推論スレッド含む）はキューへ積むだけで即座に戻り、
        // 実際のファイルI/Oはバックグラウンドの専用タスクが引き受ける（開閉コストも呼び出し毎に発生させない）。
        private readonly Channel<string> _logChannel = Channel.CreateUnbounded<string>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });
        private StreamWriter? _logWriter;
        private Task? _logWriterTask;

        private AppConfig _config = new();
        private CancellationTokenSource? _cts;
        private Task? _currentInferenceTask;

        // InitializeComponent()の実行中は、Slider等のXAML宣言済みValue（例:TileBatchSizeSlider Value="35"）が
        // 既定値から変わることでValueChangedが「ユーザー操作ではなく初期化として」発火し、そこから
        // SaveConfig()が呼ばれることがある。この時点ではXAML後方に書かれた兄弟コントロール（例：
        // OnPrefetchCountChanged由来のSaveConfig()がTileBatchSizeSliderを参照する等）がまだ接続されておらず
        // NullReferenceException（"Config Save Error"としてログに出るだけで実害はない）の原因になっていたため、
        // InitializeComponent()完了までの間はSaveConfig()を無視するガードを設ける。
        private bool _isInitializingUi = true;

        public MainWindow(string[]? args = null)
        {
            InitializeComponent();
            _isInitializingUi = false;

            // ログ出力先（カスタム指定）を反映できるよう、先に設定を読み込んでからログ基盤を初期化する
            LoadConfig();
            _logPath = ResolveLogPath(_config.LogDirectory);
            InitLog();
            WriteLog("ZiView Engine Starting (Native Core Standard)...");

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

            this.Loaded += async (s, e) =>
            {
                ScanOnnxModels();

                // AI推論が無効設定の場合、起動時のTensorRT構築・ウォームアップ自体を
                // スキップする。有効時のみ従来通りRAMDISK復元→初期化を行う
                if (_config.EnableAiInference)
                {
                    // PC再起動等でRAMDISKが消えていた場合、SSD側の退避キャッシュから復元する
                    // （TensorRT再ビルドの数十秒コストを避けるため。RAMDISK運用でない場合は自動的に何もしない）
                    RestoreTrtCacheFromBackupIfNeeded();
                    // TensorRT構築・ウォームアップは重いため、オーバーレイ表示＋バックグラウンド実行にして
                    // UIスレッドの応答なし（白画面）状態を回避する
                    await InitializeAiWithOverlayAsync(_config.SelectedModel);
                }
                ApplyConfigToUi();
                if (!string.IsNullOrEmpty(_currentSourcePath))
                {
                    _ = LoadSource(_currentSourcePath);
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

        /// <summary>
        /// ログ出力先を決定する。customDir（設定ウィンドウでの指定）があれば最優先（書き込み確認込み）。
        /// 未指定時はX:ドライブ（RAMDISK運用を想定）が存在し実際に書き込める場合は
        /// X:\temp\ZView\session.log を使い、SSDへの書き込みを避ける。それ以外は従来通り
        /// プログラムルート直下へフォールバックする。
        /// </summary>
        private static string ResolveLogPath(string? customDir)
        {
            if (!string.IsNullOrWhiteSpace(customDir))
            {
                try
                {
                    Directory.CreateDirectory(customDir);
                    string candidate = Path.Combine(customDir, "session.log");
                    File.WriteAllText(candidate, string.Empty); // 実際に書き込めるかをここで確認する
                    return candidate;
                }
                catch
                {
                    // 指定フォルダに書き込めない場合は自動判定へフォールバックする
                }
            }

            const string ramdiskLogDir = @"X:\temp\ZView";
            try
            {
                if (Directory.Exists(@"X:\"))
                {
                    Directory.CreateDirectory(ramdiskLogDir);
                    string candidate = Path.Combine(ramdiskLogDir, "session.log");
                    File.WriteAllText(candidate, string.Empty); // 実際に書き込めるかをここで確認する
                    return candidate;
                }
            }
            catch
            {
                // X:ドライブが存在しない/読み取り専用/権限不足等は、プログラムルートへフォールバックする
            }
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "session.log");
        }

        private void InitLog()
        {
            try
            {
                _logWriter = new StreamWriter(_logPath, append: false, System.Text.Encoding.UTF8) { AutoFlush = false };
                _logWriter.WriteLine($"=== Session Started at {DateTime.Now} === (log path: {_logPath})");
                _logWriter.Flush();
            }
            catch { _logWriter = null; }

            // バックグラウンドの専用タスクが1本のStreamWriterを使い回して書き続ける。
            // ファイルの開閉は起動時と終了時の1回ずつだけで、呼び出し側は完全にノンブロッキング。
            _logWriterTask = Task.Run(async () =>
            {
                try
                {
                    await foreach (var line in _logChannel.Reader.ReadAllAsync())
                    {
                        try
                        {
                            _logWriter?.WriteLine(line);
                            _logWriter?.Flush();
                        }
                        catch { /* 個々の書き込み失敗でログ基盤全体を止めない */ }
                    }
                }
                catch { /* チャンネル異常終了時も無視（アプリ終了時など） */ }
            });
        }

        private void WriteLog(string message)
        {
            // ファイルI/Oは一切ここで行わない。キューへ積むだけなのでAI推論スレッドを塞がない。
            _logChannel.Writer.TryWrite($"[{DateTime.Now:HH:mm:ss}] {message}");
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
            if (_isInitializingUi) return; // XAML解析中（InitializeComponent中）の初期値セットによる誤発火を無視
            try
            {
                _config.WindowLeft = this.Left;
                _config.WindowTop = this.Top;
                _config.WindowWidth = this.ActualWidth;
                _config.WindowHeight = this.ActualHeight;
                //_config.CheckSpread = CheckSpread.IsChecked ?? true;
                _config.CheckSpread = ReadingModeControl.Mode == PageOpenMode.RightOpen;
                _config.CheckAutoDetect = CheckAutoDetect.IsChecked ?? false;
                _config.CheckPrefetch = CheckPrefetch.IsChecked ?? true;
                _config.PrefetchPageCount = (int)PrefetchCountSlider.Value;
                _config.EnableAiInference = CheckAiEnable.IsChecked ?? true;
                _config.EnableTileBatching = CheckTileBatching.IsChecked ?? false;
                _config.TileBatchSize = (int)TileBatchSizeSlider.Value;
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

        // 後始末を完了させてから自分でCloseし直すためのフラグ（二重実行防止）
        private bool _isClosingConfirmed = false;

        private async void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            if (_isClosingConfirmed) return; // 後始末後の2回目のClose()呼び出しはそのまま閉じさせる

            e.Cancel = true; // 実行中のAI推論を安全に止めるまで、一旦ウィンドウは閉じさせない

            // 後始末中もウィンドウが「応答なし」表示にならないよう、
            // ユーザーからは即座に消えたように見せる。後始末自体はこの裏で継続する）
            Hide();

            SaveConfig();
            WriteLog("Cleaning up resources.");
            var swClose = System.Diagnostics.Stopwatch.StartNew();

            // 実行中のAI推論を止め、_onnxSessionへのアクセスが完全に終わるまで待ってからDisposeする。
            // ApplyModelSelectionAsyncと同じパターン。ONNX RuntimeのSessionはRun()実行中に
            // Dispose()すると内部でその完了を待たされるため、ここで先に止めておかないと
            // 終了処理がAI推論の残りタイル分だけ丸ごと遅延する
            _cts?.Cancel();
            var runningTask = _currentInferenceTask;
            if (runningTask != null)
            {
                try { await runningTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }
            WriteLog($"[Shutdown] Pending inference drained at {swClose.Elapsed.TotalMilliseconds:F0}ms.");　// トレース1

            // RAMDISK運用時のみ、正常終了時にSSD側へtrt_cacheを退避する
            BackupTrtCacheToSsd();
            WriteLog($"[Shutdown] BackupTrtCacheToSsd done at {swClose.Elapsed.TotalMilliseconds:F0}ms.");　// トレース2

            ClearPageCache();
            WriteLog($"[Shutdown] ClearPageCache done at {swClose.Elapsed.TotalMilliseconds:F0}ms.");　// トレース3

            CloseOpenZip();
            WriteLog($"[Shutdown] CloseOpenZip done at {swClose.Elapsed.TotalMilliseconds:F0}ms.");　// トレース4

            _onnxSession?.Dispose();
            WriteLog($"[Shutdown] _onnxSession.Dispose done at {swClose.Elapsed.TotalMilliseconds:F0}ms.");　// トレース5

            _currentCombinedOriginal?.Dispose();
            _currentCombinedUpscaled?.Dispose();

            try
            {
                if (Directory.Exists(_tempExtractDir))
                {
                    var fileCount = Directory.EnumerateFiles(_tempExtractDir, "*", SearchOption.AllDirectories).Count();
                    WriteLog($"[Shutdown] Deleting temp extract dir ({fileCount} files) at {swClose.Elapsed.TotalMilliseconds:F0}ms..."); // トレース6

                    Directory.Delete(_tempExtractDir, true);

                    WriteLog($"[Shutdown] Temp extract dir deleted at {swClose.Elapsed.TotalMilliseconds:F0}ms."); // トレース7
                }
            }
            catch (Exception exDel)
            {
                WriteLog($"[Shutdown] Temp extract dir delete failed at {swClose.Elapsed.TotalMilliseconds:F0}ms: {exDel.Message}"); // トレース8
            }

            // ログキューを締め切り、バックグラウンドタスクが残りを書き切るのを少し待ってから閉じる
            _logChannel.Writer.TryComplete();
            try { _logWriterTask?.Wait(500); } catch { }
            try { _logWriter?.Flush(); _logWriter?.Dispose(); } catch { }

            try
            {
                File.AppendAllText(_logPath,
                    $"[{DateTime.Now:HH:mm:ss}] [Shutdown] Log flush done at {swClose.Elapsed.TotalMilliseconds:F0}ms. Calling Close().{Environment.NewLine}"); // トレース9
            }
            catch { }
            // _isClosingConfirmed = true;
            // Close();
            // Close()経由の正規終了フローだと、この後CUDA/TensorRTのネイティブ側コンテキスト解放を
            // OSがブロッキングで待つため体感20〜30秒「応答なし」になる。自前の後始末は上ですべて
            // 完了済み（ログ・キャッシュ退避も終わっている）ので、ここから先はOSに丸ごと回収させる
            // 前提でプロセスを即終了する。
            Environment.Exit(0);
        }

        private void ApplyConfigToUi()
        {
            //CheckSpread.IsChecked = _config.CheckSpread;
            //ReadingModeControl.Mode = _config.CheckSpread ? PageOpenMode.RightOpen : PageOpenMode.Single;
            // 起動時の初期設定反映では、RefreshDisplay（再描画）を伴うイベントハンドラを直接呼ばず、
            // 変数とUIコントロールの見た目（Mode）の同期だけに留める
            _readingMode = _config.CheckSpread ? PageOpenMode.RightOpen : PageOpenMode.Single;
            if (ReadingModeControl != null)
            {
                ReadingModeControl.Mode = _readingMode;
            }
            CheckAutoDetect.IsChecked = _config.CheckAutoDetect;
            CheckPrefetch.IsChecked = _config.CheckPrefetch;
            SplitSlider.Value = _config.SplitSliderValue;

            CheckAiEnable.IsChecked = _config.EnableAiInference;

            int prefetchCount = Math.Clamp(_config.PrefetchPageCount, 1, 5);
            PrefetchCountSlider.Value = prefetchCount;
            PrefetchCountText.Text = $"{prefetchCount}ページ先読み";

            CheckTileBatching.IsChecked = _config.EnableTileBatching;
            int tileBatchSize = Math.Clamp(_config.TileBatchSize, 1, 64);
            TileBatchSizeSlider.Value = tileBatchSize;
            TileBatchSizeText.Text = $"{tileBatchSize}枚";

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

        private void OnTileBatchingChanged(object sender, RoutedEventArgs e) => SaveConfig();

        private void OnTileBatchSizeChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TileBatchSizeText != null) TileBatchSizeText.Text = $"{(int)e.NewValue}枚";
            SaveConfig();
        }

        private void SetupEvents()
        {
            this.MouseRightButtonUp += (s, e) => ShowContextMenu();

            this.Drop += (s, e) =>
            {
                string[]? files = e.Data.GetData(DataFormats.FileDrop) as string[];
                if (files?.Length > 0) _ = LoadSource(files[0]);
            };

            PageSlider.ValueChanged += (s, e) =>
            {
                if (_isUpdatingPageSliderInternal) return;
                RefreshDisplay();
            };

            // ホバー解除・ドラッグ解放の両方が揃った時点（＝完全に操作が終わった時点）で
            // 現在ページをAI推論付きで改めて表示し直す。操作中に抑止していた分を1回だけ回収する形。
            PageSlider.MouseLeave += (s, e) =>
            {
                if (!PageSlider.IsMouseCaptureWithin) RefreshDisplay();
            };
            PageSlider.LostMouseCapture += (s, e) =>
            {
                if (!PageSlider.IsMouseOver) RefreshDisplay();
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
