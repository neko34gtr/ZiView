using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: マウス/キーボード操作、ズーム・パン、右クリックメニュー、
    /// サイドバー/ボトムバーのアニメーション、通知バッジ、メモリ・GPU表示を担当する。
    /// </summary>
    public partial class MainWindow
    {
        // --- NVIDIA NVML P/Invoke Definitions ---
        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlInit();

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlShutdown();

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlDeviceGetHandleByIndex(uint index, out IntPtr device);

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlDeviceGetMemoryInfo(IntPtr device, out NVML_MEMORY memory);

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlDeviceGetUtilizationRates(IntPtr device, out NVML_UTILIZATION utilization);

        [StructLayout(LayoutKind.Sequential)]
        private struct NVML_MEMORY
        {
            public ulong total;
            public ulong free;
            public ulong used;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct NVML_UTILIZATION
        {
            public uint gpu;
            public uint memory;
        }

        private bool _isNvidiaGpu = false;
        private bool _nvmlInitialized = false;
        private bool _isGpuCheckAttempted = false;
        private IntPtr _nvmlDevice = IntPtr.Zero;

        // UIアニメーション・インタラクション状態管理
        private bool _isSidebarOpen = false;
        private bool _isBottomBarOpen = false;
        private System.Windows.Point _startPoint;
        private System.Windows.Point _origin;
        private bool _isDragging;

        // 表示回転（Rキー）: 設定として永続化はせず、アプリ再起動または別ソース読込でリセットされる
        private System.Windows.Media.RotateTransform? _imgRotate;
        private int _rotationAngle = 0; // 0-3（90度単位）

        private void InitGpuMonitor()
        {
            if (_isGpuCheckAttempted) return;
            _isGpuCheckAttempted = true;

            try
            {
                int result = nvmlInit();
                if (result == 0) // NVML_SUCCESS
                {
                    result = nvmlDeviceGetHandleByIndex(0, out _nvmlDevice);
                    if (result == 0)
                    {
                        _isNvidiaGpu = true;
                        _nvmlInitialized = true;
                        return;
                    }
                }
            }
            catch
            {
                // 失敗時はN/Aにフォールバック
            }

            _isNvidiaGpu = false;
            _nvmlInitialized = false;
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
                case Key.L:
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
                case Key.R:
                    RotateView();
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

        private void OnSettingChanged(object sender, RoutedEventArgs e)
        {
            // 見開き/自動判定はページの合成結果そのものを変えるため、先読み済みキャッシュは無効
            ClearPageCache();
            RefreshDisplay();
        }

        private void OnAiEnableChanged(object sender, RoutedEventArgs e)
        {
            _config.EnableAiInference = CheckAiEnable.IsChecked ?? true;
            SaveConfig();
            ClearPageCache(); // ON/OFFでキャッシュ内容（AI適用有無）の前提が変わるため破棄する

            if (!_config.EnableAiInference)
            {
                // 実行中の推論を止め、即座に原寸画像へ戻す（再起動不要）
                _cts?.Cancel();
                _currentCombinedUpscaled?.Dispose();
                _currentCombinedUpscaled = null;
                StatusText.Text = "Mode: AI Disabled";
                UpdateImageDisplay();
            }
            else if (_imageList.Count > 0)
            {
                // OFF→ONへ戻した際は現在ページを再変換して即座に反映する
                RefreshDisplay();
            }
        }

        private void OnPrefetchToggleChanged(object sender, RoutedEventArgs e)
        {
            _config.CheckPrefetch = CheckPrefetch.IsChecked ?? false;
            SaveConfig();
            if (!_config.CheckPrefetch) ClearPageCache(); // OFF時は保持している先読み分を即座に解放する
        }

        private void OnPrefetchCountChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // Minimum=1に対し既定Value=0からの自動補正で、InitializeComponent中（PrefetchCountText生成前）に
            // ValueChangedが先行発火することがあるため、未初期化の間は何もしない
            if (PrefetchCountText == null) return;

            int count = (int)PrefetchCountSlider.Value;
            _config.PrefetchPageCount = count;
            PrefetchCountText.Text = $"{count}ページ先読み";
            SaveConfig();
            ClearPageCache(); // 先読み件数の変更に伴い、既存キャッシュの前提（件数）が崩れるため破棄する
        }

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

            var itemSettings = new System.Windows.Controls.MenuItem { Header = "設定..." };
            itemSettings.Click += (s, ev) => OpenSettingsWindow();
            menu.Items.Add(itemSettings);

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

        /// <summary>
        /// MainImage.RenderTransformへ回転専用のRotateTransformを1度だけ組み込む。
        /// 既存のImgScale/ImgTranslate（ズーム・パン用）より先に適用されるよう先頭へ挿入することで、
        /// 画像自身の中心を軸にした回転→既存のズーム・パンの順で合成される。
        /// </summary>
        private void EnsureRotateTransform()
        {
            if (_imgRotate != null) return;
            _imgRotate = new System.Windows.Media.RotateTransform(0);

            if (MainImage.RenderTransform is System.Windows.Media.TransformGroup group)
            {
                group.Children.Insert(0, _imgRotate);
            }
            else
            {
                var newGroup = new System.Windows.Media.TransformGroup();
                newGroup.Children.Add(_imgRotate);
                if (MainImage.RenderTransform != null)
                    newGroup.Children.Add(MainImage.RenderTransform);
                MainImage.RenderTransform = newGroup;
            }
        }

        /// <summary>
        /// Rキー/コンテキストメニューから呼ばれる表示回転。押すたびに右90度ずつ進み、4回で元に戻る。
        /// 見開き表示中に呼ばれた場合は見開きをOFFにしてから回転する（基準は右側＝現在のPageSlider.Valueのページ）。
        /// 見開きOFF自体はOnSettingChanged経由でConfig非保存のまま反映されるため、再起動すれば見開き設定は元に戻る。
        /// 回転自体も設定として永続化しない（アプリ再起動でリセット、別ソース読込でもResetTransformによりリセットされる）。
        /// </summary>
        private void RotateView()
        {
            if (CheckSpread.IsChecked == true)
            {
                CheckSpread.IsChecked = false;
            }

            EnsureRotateTransform();
            _rotationAngle = (_rotationAngle + 1) % 4;

            _imgRotate!.CenterX = MainImage.ActualWidth / 2;
            _imgRotate.CenterY = MainImage.ActualHeight / 2;
            _imgRotate.Angle = _rotationAngle * 90;

            ShowNotification($"回転: {_rotationAngle * 90}°");
        }

        private void ShowContextMenu()
        {
            var menu = new System.Windows.Controls.ContextMenu();

            var itemFullscreen = new System.Windows.Controls.MenuItem { Header = "全画面表示の切り替え" };
            itemFullscreen.Click += (s, ev) => ToggleFullscreen();
            menu.Items.Add(itemFullscreen);

            var itemRotate = new System.Windows.Controls.MenuItem { Header = "表示を90度回転 (R)" };
            itemRotate.Click += (s, ev) => RotateView();
            menu.Items.Add(itemRotate);

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

            var itemSettings = new System.Windows.Controls.MenuItem { Header = "設定..." };
            itemSettings.Click += (s, ev) => OpenSettingsWindow();
            menu.Items.Add(itemSettings);

            menu.Items.Add(new System.Windows.Controls.Separator());

            var itemExit = new System.Windows.Controls.MenuItem { Header = "アプリケーションを終了" };
            itemExit.Click += (s, ev) => Application.Current.Shutdown();
            menu.Items.Add(itemExit);

            menu.IsOpen = true;
        }

        /// <summary>
        /// AIモデル管理・推論エンジン設定を行う専用ウィンドウを開く。
        /// 「保存して閉じる」で確定した場合のみ、モデル一覧・現在のモデルを再構築する。
        /// </summary>
        private async void OpenSettingsWindow()
        {
            var settingsWindow = new SettingsWindow(_config) { Owner = this };
            bool? result = settingsWindow.ShowDialog();

            if (result == true && settingsWindow.SettingsChanged)
            {
                SaveConfig();
                WriteLog($"Settings updated. EnginePreference={_config.EnginePreference}, ExcludedModels={_config.ExcludedModels.Count}");

                ScanOnnxModels();

                // 現在使用中のモデルが除外された可能性があるため、選択中モデルへ切り替え直す
                if (ModelComboBox.SelectedItem is string currentModel)
                {
                    await ApplyModelSelectionAsync(currentModel);
                }
                else
                {
                    // エンジン設定のみの変更でモデル一覧に変化がない場合も、
                    // エンジン優先モードの変更を反映するため再初期化する
                    _onnxSession?.Dispose();
                    _onnxSession = null;
                    _inputName = null;
                    InitializeAi();
                    if (_imageList.Count > 0) RefreshDisplay();
                }
            }
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

        private void UpdateMemoryUsage()
        {
            // 初回実行時にNVMLモニターの初期化を自動で行う（他ファイルの呼び出し順に依存させない）
            if (!_isGpuCheckAttempted)
            {
                InitGpuMonitor();
            }

            // 既存のRAM取得処理
            long workingSet = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64;
            MemoryText.Text = $"RAM: {workingSet / 1024 / 1024} MB";

            // 判定したフラグに応じてタイマー側で高速分岐・取得
            if (_isNvidiaGpu && _nvmlInitialized)
            {
                try
                {
                    if (nvmlDeviceGetUtilizationRates(_nvmlDevice, out var util) == 0 &&
                        nvmlDeviceGetMemoryInfo(_nvmlDevice, out var mem) == 0)
                    {
                        double usedMb = (double)mem.used / (1024 * 1024);
                        double totalMb = (double)mem.total / (1024 * 1024);
                        GpuText.Text = $"GPU: {util.gpu}% (VRAM: {usedMb:F0} / {totalMb:F0} MB)";
                    }
                    else
                    {
                        GpuText.Text = "GPU: Error";
                    }
                }
                catch
                {
                    GpuText.Text = "GPU: N/A";
                }
            }
            else
            {
                // 汎用GPUまたはNVIDIA以外の場合のフォールバック表示
                GpuText.Text = "GPU: N/A (Generic)";
            }
        }

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
}
