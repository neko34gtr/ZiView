using System;
using System.Runtime.InteropServices;
using System.Windows;

using OpenCvSharp;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: トーンカーブ調整機能（Tキー）を担当する。
    /// オーバーレイパネル（ToneCurveOverlay）のON/OFF管理と、確定したLUT(byte[256])を
    /// 表示直前のMatへCv2.LUTで適用する非破壊フィルタ（ApplyToneCurveForDisplay）を持つ。
    /// Mat本体やページキャッシュには一切書き込まない。
    /// </summary>
    public partial class MainWindow
    {
        // トーンカーブ調整（Tキー）: オーバーレイパネルのON/OFFとは別に、適用中のLUTはパネルを閉じても保持される。
        // Mat本体やページキャッシュには一切書き込まず、表示直前（Display.cs側）にのみ適用する非破壊フィルタ。
        private ToneCurveOverlay? _toneCurveOverlay;
        private byte[]? _toneCurveLut; // null=無調整（Display側で分岐しコピーコストを避ける）

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
        /// トーンカーブ(Tキー)のLUTが設定されている場合、表示直前のMatへLUTを適用した新規Matを返す。
        /// srcそのものは一切変更しない（_currentCombinedOriginal/_currentCombinedUpscaled等の
        /// キャッシュ済みMatを汚さないため、必ず新規Matとして返す）。未設定時はsrcをそのまま返し、無駄なコピーを避ける。
        /// </summary>
        private Mat ApplyToneCurveForDisplay(Mat src)
        {
            if (_toneCurveLut == null) return src;

            using var lutMat = new Mat(1, 256, MatType.CV_8UC1);
            Marshal.Copy(_toneCurveLut, 0, lutMat.Data, 256);

            var dst = new Mat();
            Cv2.LUT(src, lutMat, dst);
            return dst;
        }
    }
}
