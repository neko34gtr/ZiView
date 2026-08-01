using System;
using System.Collections.Generic;
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

        // トーンカーブ調整（Tキー）: オーバーレイパネルのON/OFFとは別に、適用中のLUTはパネルを閉じても保持される。
        // Mat本体やページキャッシュには一切書き込まず、表示直前（Display.cs側）にのみ適用する非破壊フィルタ。
        private ToneCurveOverlay? _toneCurveOverlay;
        private byte[]? _toneCurveLut; // null=無調整（Display側で分岐しコピーコストを避ける）

        // 部分拡大ルーペ（Mキー）: MainImage.Sourceのビットマップから該当範囲を切り出して拡大表示する。
        // 表示中のみ+/-キーで倍率を一時調整でき、ホイールでも操作可能（Shift+ホイールでルーペ枠のサイズ、
        // ホイール単体で倍率）。いずれも永続化しない。回転中は切り出した範囲を同じ角度だけ回転させ、実際の画面表示と一致させる。
        private System.Windows.Controls.Border? _loupeOverlay;
        private System.Windows.Controls.Image? _loupeImage;
        private bool _loupeModeActive = false;
        private double _loupeDisplaySize = 220.0;
        private const double MinLoupeDisplaySize = 120.0;
        private const double MaxLoupeDisplaySize = 600.0;
        private const double LoupeSizeStep = 20.0; // Shift+ホイール1ノッチあたりの増減量
        private double _loupeMagnification = 1.6;
        private const double MinLoupeMagnification = 1.0;
        private const double MaxLoupeMagnification = 8.0;
        private const double LoupeMagnificationKeyStep = 0.2;   // +/-キー1回あたりの増減量
        private const double LoupeMagnificationWheelStep = 0.1; // ホイール1ノッチあたりの増減量
        private System.Windows.Point? _lastLoupeRootGridPos;
        private System.Windows.Point? _lastLoupeMainImagePos;

        // 見開き時の片側ページ位置微調整（Shift+ドラッグ）。表示直前のTransform/Canvas座標のみを動的に
        // 変更する非破壊な実装で、元画像ファイル・_currentCombinedOriginal等のキャッシュ生データ・
        // 回転(LayoutTransform)/トーンカーブ/ルーペの各処理系には一切干渉しない。
        // オフセットは常にそのページ自身の実ピクセル単位（表示倍率非依存）で保持するため、
        // 将来的にファイルキー単位でSQLite等へそのまま永続化できる形になっている（今回はメモリ内Dictionaryのみ）。
        private readonly Dictionary<string, System.Windows.Point> _pageOffsets = new();

        private System.Windows.Controls.Canvas? _pageOffsetOverlayCanvas;

        private sealed class PageOffsetOverlayUnit
        {
            public System.Windows.Shapes.Rectangle Mask = null!;
            public System.Windows.Controls.Image CropImage = null!;
        }
        private PageOffsetOverlayUnit? _leftOffsetUnit;
        private PageOffsetOverlayUnit? _rightOffsetUnit;

        private bool _isPageOffsetDragging = false;
        private bool _pageOffsetDragIsLeft;
        private string? _pageOffsetDragKey;
        private Rect _pageOffsetDragRect;
        private double _pageOffsetDragFitScale;
        private System.Windows.Point _pageOffsetDragStartMouseLocal;
        private System.Windows.Point _pageOffsetDragStartOffset;
        private System.Windows.Point _pageOffsetDragLiveOffset;

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

            // Shift+ドラッグは見開き時の片側ページ位置微調整、それ以外は従来通りの全体パン
            if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift) && TryBeginPageOffsetDrag(e))
            {
                return;
            }

            _isDragging = true;
            _startPoint = e.GetPosition(RootGrid);
            _origin = new System.Windows.Point(ImgTranslate.X, ImgTranslate.Y);
            ImageContainer.CaptureMouse();
        }

        private void ImageContainer_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (_isPageOffsetDragging)
            {
                EndPageOffsetDrag();
                ImageContainer.ReleaseMouseCapture();
                return;
            }

            _isDragging = false;
            ImageContainer.ReleaseMouseCapture();
        }

        private void ImageContainer_MouseMove(object sender, MouseEventArgs e)
        {
            // ルーペモード中はドラッグの有無に関わらずカーソル追従だけ行う（パン処理とは独立、既存のパン挙動には無関係）
            if (_loupeModeActive)
            {
                UpdateLoupePosition(e.GetPosition(RootGrid), e.GetPosition(MainImage));
            }

            if (_isPageOffsetDragging)
            {
                UpdatePageOffsetDrag(e.GetPosition(MainImage));
                return;
            }

            if (!_isDragging) return;
            System.Windows.Point currentPoint = e.GetPosition(RootGrid);
            ImgTranslate.X = _origin.X + (currentPoint.X - _startPoint.X);
            ImgTranslate.Y = _origin.Y + (currentPoint.Y - _startPoint.Y);
        }

        private void ImageContainer_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (MainImage.Source == null) return;

            if (_loupeModeActive)
            {
                // ルーペ表示中はホイールをページ送り/ズームではなくルーペ自体の操作に割り当てる。
                // Shift+ホイール＝ルーペ枠のサイズ変更（「左右」に相当する操作をShift+ホイールで代用）、
                // ホイール単体＝拡大率変更。実ハードウェアの水平ホイール(WM_MOUSEHWHEEL)はWPF標準イベントでは
                // 拾えないため、多くのアプリで採用されているShift+縦ホイール＝横方向操作の慣例に合わせている。
                if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
                {
                    AdjustLoupeSize(e.Delta > 0 ? LoupeSizeStep : -LoupeSizeStep);
                }
                else
                {
                    AdjustLoupeMagnification(e.Delta > 0 ? LoupeMagnificationWheelStep : -LoupeMagnificationWheelStep);
                }
                return;
            }

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
                case Key.T:
                    ToggleToneCurveOverlay();
                    e.Handled = true;
                    break;
                case Key.M:
                    ToggleLoupeMode();
                    e.Handled = true;
                    break;
                case Key.OemPlus:
                case Key.Add:
                    if (_loupeModeActive)
                    {
                        AdjustLoupeMagnification(LoupeMagnificationKeyStep);
                        e.Handled = true;
                    }
                    break;
                case Key.OemMinus:
                case Key.Subtract:
                    if (_loupeModeActive)
                    {
                        AdjustLoupeMagnification(-LoupeMagnificationKeyStep);
                        e.Handled = true;
                    }
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
        /// MainImage.LayoutTransformへ回転専用のRotateTransformを1度だけ組み込む。
        /// ズーム・パン用のImgScale/ImgTranslateはRenderTransform（レイアウト確定後に描画だけをずらす）のままにし、
        /// 回転はあえてLayoutTransform側に持たせる。LayoutTransformは測定・配置(レイアウト)の段階で効くため、
        /// 90/270度回転で縦横が入れ替わった場合でもWPFが自動的にMainImageのレイアウト枠を再計算し、
        /// Stretch=Uniformと合わせてウィンドウ（コンテナ）に収まるサイズへ自動フィットする。
        /// RenderTransformのままだと枠は元の縦横のまま変わらず、回転後の画像だけがはみ出す/隙間ができる問題があったための対応。
        /// </summary>
        private void EnsureRotateTransform()
        {
            if (_imgRotate != null) return;
            _imgRotate = new System.Windows.Media.RotateTransform(0);
            MainImage.LayoutTransform = _imgRotate;
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
            _imgRotate!.Angle = _rotationAngle * 90;

            ShowNotification($"回転: {_rotationAngle * 90}°");
        }

        /// <summary>
        /// トーンカーブオーバーレイをRootGrid上へ1度だけ生成し配置する。既存XAMLは一切変更せず、
        /// コードビハインドから動的に追加する（RootGridの行/列定義数に合わせてSpanを設定し、既存レイアウトを崩さない）。
        /// CurveChangedはLUT(byte[256])を都度受け取り、Mat側は一切書き換えずUpdateImageDisplay側で表示直前に適用させる。
        /// </summary>
        private void EnsureToneCurveOverlay()
        {
            if (_toneCurveOverlay != null) return;

            _toneCurveOverlay = new ToneCurveOverlay
            {
                HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 60, 20, 0),
                Visibility = Visibility.Collapsed
            };
            _toneCurveOverlay.CurveChanged += (s, lut) =>
            {
                // リセット直後など恒等LUT（無調整）の場合はnullにしておき、
                // Display.cs側のApplyToneCurveForDisplayが完全に処理をスキップする高速パスへ入るようにする
                _toneCurveLut = IsIdentityLut(lut) ? null : lut;
                UpdateImageDisplay();
            };

            System.Windows.Controls.Grid.SetRowSpan(_toneCurveOverlay, Math.Max(1, RootGrid.RowDefinitions.Count));
            System.Windows.Controls.Grid.SetColumnSpan(_toneCurveOverlay, Math.Max(1, RootGrid.ColumnDefinitions.Count));
            System.Windows.Controls.Panel.SetZIndex(_toneCurveOverlay, 500);
            RootGrid.Children.Add(_toneCurveOverlay);
        }

        /// <summary>
        /// Tキー/コンテキストメニューから呼ばれる。パネルの表示/非表示だけを切り替える。
        /// 一度調整したLUTはパネルを閉じても_toneCurveLutに残り、画像へ適用され続ける（Photoshop等のトーンカーブと同様の挙動）。
        /// </summary>
        private void ToggleToneCurveOverlay()
        {
            EnsureToneCurveOverlay();
            bool show = _toneCurveOverlay!.Visibility != Visibility.Visible;
            _toneCurveOverlay.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
            ShowNotification(show ? "トーンカーブ: 表示" : "トーンカーブ: 非表示");
        }

        private static bool IsIdentityLut(byte[] lut)
        {
            for (int i = 0; i < 256; i++)
            {
                if (lut[i] != i) return false;
            }
            return true;
        }

        /// <summary>
        /// ルーペオーバーレイをRootGrid上へ1度だけ生成する。
        /// 以前はVisualBrush＋Viewboxで対象要素の見た目をそのまま複製する方式だったが、
        /// 「対象要素のローカル座標系がどこを基準にしているか」がWPF内部の変形合成に依存し、
        /// 見開きON/OFF（合成後の画像サイズが変わる）や回転（LayoutTransform）で拡大範囲が大きくズレる問題があった。
        /// 本実装ではその曖昧さを排除し、MainImage.Source（今まさに画面に出ているビットマップそのもの、
        /// 見開き合成・AI超解像・トーンカーブ適用済み）から、カーソル位置に対応する実ピクセル範囲をCroppedBitmapで
        /// 直接切り出して表示する。カーソル→ビットマップ座標の変換はMainImage自身のStretch=Uniformによる
        /// レターボックスを自前で計算するだけなので、画像サイズ（見開き有無）や向き（回転）が変わっても
        /// 常に「今表示されているビットマップのどのピクセルの上にカーソルがあるか」を正しく求められる。
        /// </summary>
        private void EnsureLoupeOverlay()
        {
            if (_loupeOverlay != null) return;

            _loupeImage = new System.Windows.Controls.Image
            {
                Stretch = System.Windows.Media.Stretch.Fill
            };
            // ピクセル単位を拡大して見るためのツールなので、補間で滲ませず最近傍でくっきり見せる
            System.Windows.Media.RenderOptions.SetBitmapScalingMode(
                _loupeImage, System.Windows.Media.BitmapScalingMode.NearestNeighbor);

            _loupeOverlay = new System.Windows.Controls.Border
            {
                Width = _loupeDisplaySize,
                Height = _loupeDisplaySize,
                BorderBrush = System.Windows.Media.Brushes.Gold,
                BorderThickness = new Thickness(2),
                Background = System.Windows.Media.Brushes.Black,
                Child = _loupeImage,
                Visibility = Visibility.Collapsed,
                IsHitTestVisible = false, // 下にある画像へのクリック/ドラッグ、およびマウス移動の継続受信を妨げない
                HorizontalAlignment = HorizontalAlignment.Left,
                VerticalAlignment = VerticalAlignment.Top
            };

            System.Windows.Controls.Grid.SetRowSpan(_loupeOverlay, Math.Max(1, RootGrid.RowDefinitions.Count));
            System.Windows.Controls.Grid.SetColumnSpan(_loupeOverlay, Math.Max(1, RootGrid.ColumnDefinitions.Count));
            System.Windows.Controls.Panel.SetZIndex(_loupeOverlay, 600);
            RootGrid.Children.Add(_loupeOverlay);
        }

        /// <summary>
        /// Mキー/コンテキストメニューから呼ばれる。ルーペモードのON/OFFだけを切り替える
        /// （実際の追従・拡大更新はImageContainer_MouseMoveから呼ばれるUpdateLoupePositionが担当）。
        /// </summary>
        private void ToggleLoupeMode()
        {
            EnsureLoupeOverlay();
            _loupeModeActive = !_loupeModeActive;
            _loupeOverlay!.Visibility = _loupeModeActive ? Visibility.Visible : Visibility.Collapsed;
            ShowNotification(_loupeModeActive ? "ルーペモード: ON" : "ルーペモード: OFF");
        }

        /// <summary>
        /// ルーペモード中、マウス移動のたびに呼ばれる。
        /// rootGridPosはルーペ本体（枠）自体をカーソル中心に配置するための位置決めに、
        /// mainImagePosはMainImage自身のローカル座標（回転が解決済みの自然な向きの座標系）で、
        /// 拡大表示すべきビットマップ上の範囲を求めるために使う。
        /// 直近の位置は_lastLoupeRootGridPos/_lastLoupeMainImagePosへキャッシュし、
        /// マウスを動かさずに+/-キーで倍率だけ変更した場合も同じ位置で即座に再描画できるようにする。
        /// </summary>
        private void UpdateLoupePosition(System.Windows.Point rootGridPos, System.Windows.Point mainImagePos)
        {
            if (_loupeOverlay == null || _loupeImage == null || !_loupeModeActive) return;
            if (MainImage.Source is not System.Windows.Media.Imaging.BitmapSource bmp) return;
            if (bmp.PixelWidth <= 0 || bmp.PixelHeight <= 0) return;

            double boxW = MainImage.ActualWidth, boxH = MainImage.ActualHeight;
            if (boxW <= 0 || boxH <= 0) return;

            _lastLoupeRootGridPos = rootGridPos;
            _lastLoupeMainImagePos = mainImagePos;

            // MainImage(Stretch=Uniform)内でSourceビットマップが実際に描画されている領域（レターボックス考慮）を逆算し、
            // カーソル位置をビットマップの実ピクセル座標へ変換する。
            double srcW = bmp.PixelWidth, srcH = bmp.PixelHeight;
            double fitScale = Math.Min(boxW / srcW, boxH / srcH);
            double dispW = srcW * fitScale, dispH = srcH * fitScale;
            double offX = (boxW - dispW) / 2.0, offY = (boxH - dispH) / 2.0;

            double bx = (mainImagePos.X - offX) / fitScale;
            double by = (mainImagePos.Y - offY) / fitScale;

            int viewPx = Math.Max(4, (int)(_loupeDisplaySize / _loupeMagnification));
            int cropX = (int)Math.Clamp(bx - viewPx / 2.0, 0, Math.Max(0, srcW - viewPx));
            int cropY = (int)Math.Clamp(by - viewPx / 2.0, 0, Math.Max(0, srcH - viewPx));
            int cropW = (int)Math.Min(viewPx, srcW);
            int cropH = (int)Math.Min(viewPx, srcH);

            if (cropW <= 0 || cropH <= 0)
            {
                _loupeImage.Source = null;
            }
            else
            {
                var cropped = new System.Windows.Media.Imaging.CroppedBitmap(
                    bmp, new Int32Rect(cropX, cropY, cropW, cropH));

                // MainImage.Sourceは回転前（自然な向き）のビットマップなので、切り出した範囲を
                // 画面表示と同じ角度だけ回転させないと、回転表示中は向きの合わない中身が出てしまう。
                _loupeImage.Source = _rotationAngle == 0
                    ? cropped
                    : new System.Windows.Media.Imaging.TransformedBitmap(
                        cropped, new System.Windows.Media.RotateTransform(_rotationAngle * 90));
            }

            // ルーペ本体はカーソル位置を中心に表示する（ウィンドウ端では内側へクランプ）
            double left = rootGridPos.X - _loupeDisplaySize / 2.0;
            double top = rootGridPos.Y - _loupeDisplaySize / 2.0;
            double maxLeft = Math.Max(0, RootGrid.ActualWidth - _loupeDisplaySize);
            double maxTop = Math.Max(0, RootGrid.ActualHeight - _loupeDisplaySize);
            left = Math.Clamp(left, 0, maxLeft);
            top = Math.Clamp(top, 0, maxTop);

            _loupeOverlay.Margin = new Thickness(left, top, 0, 0);
        }

        /// <summary>
        /// Shift+ホイールから呼ばれる。ルーペ枠自体の表示サイズ（一辺の長さ）を一時的に変更する。
        /// 拡大率とは独立したパラメータで、こちらもconfig.jsonへは保存しない。
        /// </summary>
        private void AdjustLoupeSize(double delta)
        {
            if (_loupeOverlay == null) return;

            double next = Math.Clamp(_loupeDisplaySize + delta, MinLoupeDisplaySize, MaxLoupeDisplaySize);
            if (Math.Abs(next - _loupeDisplaySize) < 0.001) return;
            _loupeDisplaySize = next;

            _loupeOverlay.Width = _loupeDisplaySize;
            _loupeOverlay.Height = _loupeDisplaySize;

            if (_lastLoupeRootGridPos.HasValue && _lastLoupeMainImagePos.HasValue)
            {
                UpdateLoupePosition(_lastLoupeRootGridPos.Value, _lastLoupeMainImagePos.Value);
            }

            ShowNotification($"ルーペサイズ: {(int)_loupeDisplaySize}px");
        }

        /// <summary>
        /// +/-キー、またはホイール単体から呼ばれる。ルーペ表示中のみ有効な一時的な倍率調整で、
        /// config.jsonへは保存しない（次回ON時は直前の倍率を引き継ぐが、あくまでアプリ内の一時状態）。
        /// マウスを動かしていなくても直近位置のキャッシュを使って即座に再描画する。
        /// </summary>
        private void AdjustLoupeMagnification(double delta)
        {
            double next = Math.Clamp(_loupeMagnification + delta, MinLoupeMagnification, MaxLoupeMagnification);
            if (Math.Abs(next - _loupeMagnification) < 0.001) return;
            _loupeMagnification = next;

            if (_lastLoupeRootGridPos.HasValue && _lastLoupeMainImagePos.HasValue)
            {
                UpdateLoupePosition(_lastLoupeRootGridPos.Value, _lastLoupeMainImagePos.Value);
            }

            ShowNotification($"ルーペ倍率: {_loupeMagnification:F1}x");
        }

        /// <summary>
        /// 現在表示中のビットマップ上で、見開きの左ページ/右ページそれぞれが実際に描画されている
        /// 矩形（MainImageのローカル座標系＝回転・ズームが解決済みの自然な向きの座標）を求める。
        /// ルーペ機能と同じくMainImage自身のStretch=Uniformレターボックスを自前計算するだけなので、
        /// 見開きON/OFFや回転の状態変化にも自動的に追従する。非見開き時や画像未読込時はfalseを返す。
        /// </summary>
        private bool TryGetPageHalfRects(out Rect leftRect, out Rect rightRect, out double fitScale)
        {
            leftRect = rightRect = Rect.Empty;
            fitScale = 0;

            if (_currentSpreadLeftWidth <= 0) return false; // 非見開き、または左ページなし
            if (MainImage.Source is not System.Windows.Media.Imaging.BitmapSource bmp) return false;
            if (bmp.PixelWidth <= 0 || bmp.PixelHeight <= 0) return false;

            double boxW = MainImage.ActualWidth, boxH = MainImage.ActualHeight;
            if (boxW <= 0 || boxH <= 0) return false;

            double srcW = bmp.PixelWidth, srcH = bmp.PixelHeight;
            fitScale = Math.Min(boxW / srcW, boxH / srcH);
            double dispW = srcW * fitScale, dispH = srcH * fitScale;
            double offX = (boxW - dispW) / 2.0, offY = (boxH - dispH) / 2.0;

            double splitScreenX = offX + _currentSpreadLeftWidth * fitScale;
            leftRect = new Rect(offX, offY, splitScreenX - offX, dispH);
            rightRect = new Rect(splitScreenX, offY, offX + dispW - splitScreenX, dispH);
            return true;
        }

        /// <summary>
        /// Shift+左クリックの押下時に呼ばれる。クリック位置が見開きの左右どちらのページ上かを判定し、
        /// 該当すればそのページの位置微調整ドラッグを開始する。対象外（非見開き・境界外クリック等）ならfalseを返し、
        /// 呼び出し側（ImageContainer_MouseLeftButtonDown）は通常の全体パンへフォールバックする。
        /// </summary>
        private bool TryBeginPageOffsetDrag(MouseButtonEventArgs e)
        {
            if (!TryGetPageHalfRects(out var leftRect, out var rightRect, out double fitScale)) return false;

            var posLocal = e.GetPosition(MainImage);
            bool isLeft;
            Rect rect;
            string? key;

            if (leftRect.Contains(posLocal)) { isLeft = true; rect = leftRect; key = _currentSpreadLeftKey; }
            else if (rightRect.Contains(posLocal)) { isLeft = false; rect = rightRect; key = _currentSpreadRightKey; }
            else return false;

            if (string.IsNullOrEmpty(key)) return false;

            _isPageOffsetDragging = true;
            _pageOffsetDragIsLeft = isLeft;
            _pageOffsetDragKey = key;
            _pageOffsetDragRect = rect;
            _pageOffsetDragFitScale = fitScale;
            _pageOffsetDragStartMouseLocal = posLocal;
            _pageOffsetDragStartOffset = _pageOffsets.TryGetValue(key, out var existing) ? existing : new System.Windows.Point(0, 0);
            _pageOffsetDragLiveOffset = _pageOffsetDragStartOffset;

            SetupPageOffsetUnit(isLeft, rect, fitScale, _pageOffsetDragStartOffset);
            ImageContainer.CaptureMouse();
            e.Handled = true;
            return true;
        }

        /// <summary>
        /// ページ位置微調整ドラッグ中、マウス移動のたびに呼ばれる。MainImageのローカル座標での移動量を
        /// 現在のfitScaleで割り戻すことで、ズーム倍率に依存しない「そのページ自身の実ピクセル単位」の
        /// オフセットへ変換する（ズーム状態やウィンドウサイズが変わっても常に同じ意味を持つ値になる）。
        /// 内容（CroppedBitmap）は再生成せず位置のみを更新するため軽量。
        /// </summary>
        private void UpdatePageOffsetDrag(System.Windows.Point mainImagePos)
        {
            if (!_isPageOffsetDragging || _pageOffsetDragFitScale <= 0) return;

            double dxLocal = mainImagePos.X - _pageOffsetDragStartMouseLocal.X;
            double dyLocal = mainImagePos.Y - _pageOffsetDragStartMouseLocal.Y;

            _pageOffsetDragLiveOffset = new System.Windows.Point(
                _pageOffsetDragStartOffset.X + dxLocal / _pageOffsetDragFitScale,
                _pageOffsetDragStartOffset.Y + dyLocal / _pageOffsetDragFitScale);

            MovePageOffsetUnit(_pageOffsetDragIsLeft, _pageOffsetDragRect, _pageOffsetDragFitScale, _pageOffsetDragLiveOffset);
        }

        /// <summary>
        /// ドラッグ終了（MouseUp）時に呼ばれる。直近のライブオフセットを確定値として_pageOffsetsへ反映する。
        /// ほぼ0（0.5px未満）まで戻された場合は「調整なし」とみなしDictionaryから削除し、
        /// 対応するオーバーレイも取り除いて既定表示（MainImageそのまま）へ戻す。
        /// </summary>
        private void EndPageOffsetDrag()
        {
            if (_pageOffsetDragKey != null)
            {
                var finalOffset = _pageOffsetDragLiveOffset;
                const double epsilon = 0.5;

                if (Math.Abs(finalOffset.X) < epsilon && Math.Abs(finalOffset.Y) < epsilon)
                {
                    _pageOffsets.Remove(_pageOffsetDragKey);
                }
                else
                {
                    _pageOffsets[_pageOffsetDragKey] = finalOffset;
                }

                ShowNotification($"ページ位置を調整: dX={finalOffset.X:F0}, dY={finalOffset.Y:F0}");
            }

            _isPageOffsetDragging = false;
            _pageOffsetDragKey = null;

            RefreshPageOffsetOverlays();
        }

        /// <summary>
        /// ページ切り替え・AI処理完了・トーンカーブ変更など、表示が更新されるたびにUpdateImageDisplayから
        /// 呼ばれる。現在表示中ページ（左右）それぞれについて、保存済みオフセットがあればオーバーレイを
        /// 表示・更新し、なければ（＝調整なし、または非見開きへ切り替わった等）オーバーレイを取り除く。
        /// ドラッグ中に呼ばれても安全（何もしない）。
        /// </summary>
        private void RefreshPageOffsetOverlays()
        {
            if (_isPageOffsetDragging) return;

            if (!TryGetPageHalfRects(out var leftRect, out var rightRect, out double fitScale))
            {
                RemovePageOffsetUnit(true);
                RemovePageOffsetUnit(false);
                return;
            }

            ApplyOrRemoveSide(true, leftRect, _currentSpreadLeftKey);
            ApplyOrRemoveSide(false, rightRect, _currentSpreadRightKey);

            void ApplyOrRemoveSide(bool isLeft, Rect rect, string? key)
            {
                if (!string.IsNullOrEmpty(key) && _pageOffsets.TryGetValue(key, out var offset))
                {
                    SetupPageOffsetUnit(isLeft, rect, fitScale, offset);
                }
                else
                {
                    RemovePageOffsetUnit(isLeft);
                }
            }
        }

        /// <summary>
        /// ページ位置微調整用のオーバーレイCanvasを1度だけ生成し、MainImageの実際の親パネルへ
        /// 兄弟要素として追加する。LayoutTransform/RenderTransformはMainImageと同一のオブジェクト参照を
        /// 共有させることで、回転・ズーム・パンに常に自動追従させる（個別に同期コードを書く必要がない）。
        /// </summary>
        private void EnsurePageOffsetOverlayCanvas()
        {
            if (_pageOffsetOverlayCanvas != null) return;

            EnsureRotateTransform(); // MainImage.LayoutTransformが必ず有効なRotateTransformである状態にしてから共有する

            _pageOffsetOverlayCanvas = new System.Windows.Controls.Canvas
            {
                IsHitTestVisible = false,
                ClipToBounds = true,
                LayoutTransform = MainImage.LayoutTransform,
                RenderTransform = MainImage.RenderTransform
            };

            if (MainImage.Parent is System.Windows.Controls.Panel parentPanel)
            {
                parentPanel.Children.Add(_pageOffsetOverlayCanvas);
            }
            System.Windows.Controls.Panel.SetZIndex(_pageOffsetOverlayCanvas, 50); // MainImageのすぐ上、他のオーバーレイ(トーンカーブ/ルーペ)よりは下
        }

        /// <summary>
        /// オーバーレイCanvas自体の箱（サイズ・配置）をMainImageの現在値へ同期する。
        /// ウィンドウサイズ変化などでMainImageの実際のレイアウト矩形が変わっている可能性があるため、
        /// コンテンツを配置する直前に毎回呼び出す。
        /// </summary>
        private void SyncPageOffsetOverlayCanvasBox()
        {
            if (_pageOffsetOverlayCanvas == null) return;
            _pageOffsetOverlayCanvas.HorizontalAlignment = MainImage.HorizontalAlignment;
            _pageOffsetOverlayCanvas.VerticalAlignment = MainImage.VerticalAlignment;
            _pageOffsetOverlayCanvas.Margin = MainImage.Margin;
            _pageOffsetOverlayCanvas.Width = MainImage.ActualWidth;
            _pageOffsetOverlayCanvas.Height = MainImage.ActualHeight;
        }

        private PageOffsetOverlayUnit CreatePageOffsetUnit()
        {
            return new PageOffsetOverlayUnit
            {
                Mask = new System.Windows.Shapes.Rectangle
                {
                    Fill = System.Windows.Media.Brushes.Black,
                    IsHitTestVisible = false
                },
                CropImage = new System.Windows.Controls.Image
                {
                    Stretch = System.Windows.Media.Stretch.Fill,
                    IsHitTestVisible = false
                }
            };
        }

        /// <summary>
        /// 指定サイドのオーバーレイ（マスク＋切り出し画像）を用意し、指定オフセットの位置へ配置する。
        /// マスクは常に元の位置に固定してMainImage側の該当範囲を覆い隠し（ズレたゴーストが見えないようにする）、
        /// 切り出し画像は「元の位置＋オフセット」へ配置する。呼び出しのたびにCroppedBitmapを取り直すため、
        /// トーンカーブ変更やAI処理完了などでMainImage.Sourceが更新された場合も内容が正しく追従する。
        /// ドラッグ開始時・ページ再表示時など「内容が変わりうるタイミング」で使うやや重い版。
        /// </summary>
        private void SetupPageOffsetUnit(bool isLeft, Rect rect, double fitScale, System.Windows.Point offsetPixels)
        {
            if (rect.Width <= 0 || rect.Height <= 0) return;
            if (MainImage.Source is not System.Windows.Media.Imaging.BitmapSource bmp) return;

            EnsurePageOffsetOverlayCanvas();
            SyncPageOffsetOverlayCanvasBox();

            var unit = isLeft
                ? (_leftOffsetUnit ??= CreatePageOffsetUnit())
                : (_rightOffsetUnit ??= CreatePageOffsetUnit());

            if (unit.Mask.Parent == null) _pageOffsetOverlayCanvas!.Children.Add(unit.Mask);
            if (unit.CropImage.Parent == null) _pageOffsetOverlayCanvas!.Children.Add(unit.CropImage);

            unit.Mask.Width = rect.Width;
            unit.Mask.Height = rect.Height;
            System.Windows.Controls.Canvas.SetLeft(unit.Mask, rect.X);
            System.Windows.Controls.Canvas.SetTop(unit.Mask, rect.Y);

            int cropX = isLeft ? 0 : _currentSpreadLeftWidth;
            int cropW = isLeft ? _currentSpreadLeftWidth : Math.Max(0, bmp.PixelWidth - _currentSpreadLeftWidth);
            int cropH = bmp.PixelHeight;
            if (cropW <= 0 || cropH <= 0) return;

            unit.CropImage.Source = new System.Windows.Media.Imaging.CroppedBitmap(bmp, new Int32Rect(cropX, 0, cropW, cropH));
            unit.CropImage.Width = rect.Width;
            unit.CropImage.Height = rect.Height;

            System.Windows.Controls.Canvas.SetLeft(unit.CropImage, rect.X + offsetPixels.X * fitScale);
            System.Windows.Controls.Canvas.SetTop(unit.CropImage, rect.Y + offsetPixels.Y * fitScale);
        }

        /// <summary>
        /// 既にSetupPageOffsetUnitで内容（CroppedBitmap）が設定済みの前提で、位置のみを更新する軽量版。
        /// ドラッグ中のMouseMoveのたびに呼ばれるため、画像処理を一切行わずCanvas.Left/Topの更新のみに留める。
        /// </summary>
        private void MovePageOffsetUnit(bool isLeft, Rect rect, double fitScale, System.Windows.Point offsetPixels)
        {
            var unit = isLeft ? _leftOffsetUnit : _rightOffsetUnit;
            if (unit == null) return;

            System.Windows.Controls.Canvas.SetLeft(unit.CropImage, rect.X + offsetPixels.X * fitScale);
            System.Windows.Controls.Canvas.SetTop(unit.CropImage, rect.Y + offsetPixels.Y * fitScale);
        }

        /// <summary>
        /// 指定サイドのオーバーレイ（マスク＋切り出し画像）を取り除き、既定表示（MainImageそのまま）へ戻す。
        /// </summary>
        private void RemovePageOffsetUnit(bool isLeft)
        {
            var unit = isLeft ? _leftOffsetUnit : _rightOffsetUnit;
            if (unit == null) return;

            _pageOffsetOverlayCanvas?.Children.Remove(unit.Mask);
            _pageOffsetOverlayCanvas?.Children.Remove(unit.CropImage);

            if (isLeft) _leftOffsetUnit = null; else _rightOffsetUnit = null;
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

            var itemToneCurve = new System.Windows.Controls.MenuItem { Header = "トーンカーブ (T)" };
            itemToneCurve.Click += (s, ev) => ToggleToneCurveOverlay();
            menu.Items.Add(itemToneCurve);

            var itemToneCurveReset = new System.Windows.Controls.MenuItem { Header = "トーンカーブをリセット" };
            itemToneCurveReset.Click += (s, ev) =>
            {
                EnsureToneCurveOverlay();
                _toneCurveOverlay!.ResetCurve();
            };
            menu.Items.Add(itemToneCurveReset);

            var itemLoupe = new System.Windows.Controls.MenuItem { Header = "部分拡大ルーペ (M)" };
            itemLoupe.Click += (s, ev) => ToggleLoupeMode();
            menu.Items.Add(itemLoupe);

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
