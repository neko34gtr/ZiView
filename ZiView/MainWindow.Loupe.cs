using System;
using System.Windows;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: 部分拡大ルーペ機能（Mキー）を担当する。
    /// MainImage.Sourceのビットマップから該当範囲をCroppedBitmapで直接切り出して拡大表示する。
    /// 表示中は+/-キー・ホイールで倍率・サイズを一時調整でき、いずれも永続化しない。
    /// 回転中は切り出した範囲を同じ角度だけ回転させ、実際の画面表示と一致させる。
    /// </summary>
    public partial class MainWindow
    {
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
    }
}
