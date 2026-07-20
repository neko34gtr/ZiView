using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: マウス/キーボード操作、ズーム・パン、右クリックメニュー、
    /// サイドバー/ボトムバーのアニメーション、通知バッジ、メモリ表示を担当する。
    /// </summary>
    public partial class MainWindow
    {
        // UIアニメーション・インタラクション状態管理
        private bool _isSidebarOpen = false;
        private bool _isBottomBarOpen = false;
        private System.Windows.Point _startPoint;
        private System.Windows.Point _origin;
        private bool _isDragging;

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
}
