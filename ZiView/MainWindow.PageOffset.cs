using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Input;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: 見開き表示時の片側ページ位置微調整機能（Shift+左クリック+ドラッグ）を担当する。
    /// マスク矩形（固定・元位置を黒で覆う）＋切り出しImage（CroppedBitmap、オフセット位置へ移動）の
    /// 2要素構成で、MainImage自身・_currentCombinedOriginal等のキャッシュ生データ・元画像ファイルには
    /// 一切干渉しない非破壊な表示層のみの実装。
    ///
    /// オーバーレイの配置はRootGrid直下に固定したCanvasへ、MainImage.TransformToVisual(RootGrid)で
    /// 求めたスクリーン座標を絶対配置する方式（ルーペ・トーンカーブと同じ実績のあるRootGrid固定パターン＋
    /// 確実な座標変換）。オフセットはそのページ自身の実ピクセル単位（表示倍率非依存）でDictionaryに保持し、
    /// 将来的にファイルキー単位でSQLite等へそのまま永続化できる形になっている（現状はメモリ内Dictionaryのみ）。
    /// </summary>
    public partial class MainWindow
    {
        // 現在表示中ページの見開き境界情報（Display.cs側のDisplayPageが表示更新のたびに設定する）。
        // LeftWidth<=0は非見開き（または左ページ無し）を意味する。LeftKey/RightKeyはそれぞれの
        // ページの_imageList上のキー（＝将来SQLiteへ保存する際のファイル単位オフセットのキーにもなる）。
        private int _currentSpreadLeftWidth = 0;
        private string? _currentSpreadLeftKey;
        private string? _currentSpreadRightKey;

        // 見開き時の片側ページ位置微調整（Shift+ドラッグ）。表示直前のTransform/Canvas座標のみを動的に
        // 変更する非破壊な実装で、元画像ファイル・_currentCombinedOriginal等のキャッシュ生データ・
        // 回転(LayoutTransform)/トーンカーブ/ルーペの各処理系には一切干渉しない。
        // オフセットは常にそのページ自身の実ピクセル単位（表示倍率非依存）で保持するため、
        // 将来的にファイルキー単位でSQLite等へそのまま永続化できる形になっている（今回はメモリ内Dictionaryのみ）。
        private readonly Dictionary<string, System.Windows.Point> _pageOffsets = new();

        private System.Windows.Controls.Canvas? _pageOffsetOverlayCanvas;
        private System.Windows.Shapes.Rectangle? _pageOffsetSelectionBorder;

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
        private Rect _pageOffsetDragLocalRect;
        private double _pageOffsetDragFitScale;
        private System.Windows.Point _pageOffsetDragStartMouseLocal;
        private System.Windows.Point _pageOffsetDragStartOffset;
        private System.Windows.Point _pageOffsetDragLiveOffset;

        /// <summary>
        /// _currentSpreadLeftWidth（DecodeCombinedPage時点＝AI超解像適用前の原寸での左ページ幅）を、
        /// 現在実際に表示されているビットマップ（AI超解像後の場合は拡大された解像度）のピクセル空間へ
        /// 換算する。原寸合計幅に対する現在のbmp幅の比率でスケールするだけなので、AI超解像の有無・倍率に
        /// 関わらず常に正しい境界位置になる。
        /// </summary>
        private double GetEffectiveSpreadLeftWidth(System.Windows.Media.Imaging.BitmapSource bmp)
        {
            double originalTotalWidth = _currentCombinedOriginal?.Width ?? 0;
            double scaleRatio = originalTotalWidth > 0 ? (bmp.PixelWidth / originalTotalWidth) : 1.0;
            return _currentSpreadLeftWidth * scaleRatio;
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

            double effectiveLeftWidth = GetEffectiveSpreadLeftWidth(bmp);
            double splitScreenX = offX + effectiveLeftWidth * fitScale;
            leftRect = new Rect(offX, offY, splitScreenX - offX, dispH);
            rightRect = new Rect(splitScreenX, offY, offX + dispW - splitScreenX, dispH);
            return true;
        }

        /// <summary>
        /// MainImageローカル座標系の矩形を、RootGrid座標系（＝オーバーレイCanvasの配置座標系）へ変換する。
        /// MainImage.TransformToVisualはWPFが実際のレイアウト・LayoutTransform（回転）・RenderTransform
        /// （ズーム/パン）をすべて考慮して計算するため、親要素の構造やMarginを自前で推測する必要がなく、
        /// 常に画面上の実際の位置と一致する（回転90/270度で矩形が90度傾いても、角を変換してから
        /// min/maxを取り直すことで軸並行な矩形として扱える）。
        /// </summary>
        private Rect LocalRectToScreen(Rect local)
        {
            var transform = MainImage.TransformToVisual(RootGrid);
            var p1 = transform.Transform(new System.Windows.Point(local.Left, local.Top));
            var p2 = transform.Transform(new System.Windows.Point(local.Right, local.Bottom));

            double x1 = Math.Min(p1.X, p2.X), x2 = Math.Max(p1.X, p2.X);
            double y1 = Math.Min(p1.Y, p2.Y), y2 = Math.Max(p1.Y, p2.Y);
            return new Rect(x1, y1, Math.Max(0, x2 - x1), Math.Max(0, y2 - y1));
        }

        /// <summary>
        /// Shift+左クリックの押下時に呼ばれる。クリック位置が見開きの左右どちらのページ上かを判定し、
        /// 該当すればそのページの位置微調整ドラッグを開始する。対象外（非見開き・境界外クリック等）ならfalseを返し、
        /// 呼び出し側（ImageContainer_MouseLeftButtonDown）は通常の全体パンへフォールバックする。
        /// ドラッグ開始と同時に、選択された側を色付き枠で囲って操作対象が視覚的に分かるようにする。
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
            _pageOffsetDragLocalRect = rect;
            _pageOffsetDragFitScale = fitScale;
            _pageOffsetDragStartMouseLocal = posLocal;
            _pageOffsetDragStartOffset = _pageOffsets.TryGetValue(key, out var existing) ? existing : new System.Windows.Point(0, 0);
            _pageOffsetDragLiveOffset = _pageOffsetDragStartOffset;

            ShowPageOffsetSelectionBorder(rect);
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

            MovePageOffsetUnit(_pageOffsetDragIsLeft, _pageOffsetDragLocalRect, _pageOffsetDragFitScale, _pageOffsetDragLiveOffset);
        }

        /// <summary>
        /// ドラッグ終了（MouseUp）時に呼ばれる。直近のライブオフセットを確定値として_pageOffsetsへ反映する。
        /// ほぼ0（0.5px未満）まで戻された場合は「調整なし」とみなしDictionaryから削除し、
        /// 対応するオーバーレイも取り除いて既定表示（MainImageそのまま）へ戻す。選択枠も非表示にする。
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

            HidePageOffsetSelectionBorder();
            RefreshPageOffsetOverlays();
        }

        /// <summary>
        /// ページ切り替え・AI処理完了・トーンカーブ変更・ズーム/パン・回転など、表示に影響する操作のたびに
        /// 呼ばれる。現在表示中ページ（左右）それぞれについて、保存済みオフセットがあればオーバーレイを
        /// 表示・更新し、なければ（＝調整なし、または非見開きへ切り替わった等）オーバーレイを取り除く。
        /// ドラッグ中に呼ばれても安全（何もしない＝ライブ更新はUpdatePageOffsetDragが別途担当）。
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
        /// ページ位置微調整用のオーバーレイCanvasを1度だけ生成する。RootGrid直下に全面スパンで固定配置し
        /// （ルーペ/トーンカーブと同じ実績のあるパターン）、中の要素はすべて画面座標（RootGrid座標系）で
        /// 絶対配置する。以前はMainImageの親を推測してMarginを複製する方式だったが、実際のレイアウト構造と
        /// 一致せず移動が画面に反映されない不具合があったため、LocalRectToScreen（TransformToVisual）による
        /// 確実な座標変換方式へ切り替えた。
        /// </summary>
        private void EnsurePageOffsetOverlayCanvas()
        {
            if (_pageOffsetOverlayCanvas != null) return;

            _pageOffsetOverlayCanvas = new System.Windows.Controls.Canvas
            {
                IsHitTestVisible = false
            };
            System.Windows.Controls.Grid.SetRowSpan(_pageOffsetOverlayCanvas, Math.Max(1, RootGrid.RowDefinitions.Count));
            System.Windows.Controls.Grid.SetColumnSpan(_pageOffsetOverlayCanvas, Math.Max(1, RootGrid.ColumnDefinitions.Count));
            System.Windows.Controls.Panel.SetZIndex(_pageOffsetOverlayCanvas, 50); // MainImageのすぐ上、他のオーバーレイ(トーンカーブ/ルーペ)よりは下
            RootGrid.Children.Add(_pageOffsetOverlayCanvas);
        }

        /// <summary>
        /// Shift+ドラッグ中、選択されている側のページ範囲を色付きの枠線で囲って表示する。
        /// ドラッグ中は元の位置（範囲）に固定表示し、「今どのページを操作しているか」を示す。
        /// </summary>
        private void ShowPageOffsetSelectionBorder(Rect localRect)
        {
            EnsurePageOffsetOverlayCanvas();

            _pageOffsetSelectionBorder ??= new System.Windows.Shapes.Rectangle
            {
                Stroke = System.Windows.Media.Brushes.Cyan,
                StrokeThickness = 3,
                Fill = System.Windows.Media.Brushes.Transparent,
                IsHitTestVisible = false
            };
            if (_pageOffsetSelectionBorder.Parent == null)
            {
                _pageOffsetOverlayCanvas!.Children.Add(_pageOffsetSelectionBorder);
            }
            System.Windows.Controls.Panel.SetZIndex(_pageOffsetSelectionBorder, 60); // マスク/切り出し画像より上

            var screen = LocalRectToScreen(localRect);
            _pageOffsetSelectionBorder.Width = screen.Width;
            _pageOffsetSelectionBorder.Height = screen.Height;
            System.Windows.Controls.Canvas.SetLeft(_pageOffsetSelectionBorder, screen.X);
            System.Windows.Controls.Canvas.SetTop(_pageOffsetSelectionBorder, screen.Y);
            _pageOffsetSelectionBorder.Visibility = Visibility.Visible;
        }

        private void HidePageOffsetSelectionBorder()
        {
            if (_pageOffsetSelectionBorder != null)
            {
                _pageOffsetSelectionBorder.Visibility = Visibility.Collapsed;
            }
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
        /// マスクは常に元の位置（ローカル矩形をそのまま画面座標へ変換した位置）に固定してMainImage側の
        /// 該当範囲を覆い隠し（ズレたゴーストが見えないようにする）、切り出し画像は「元の位置＋オフセット」を
        /// ローカル座標で計算してから画面座標へ変換して配置する。呼び出しのたびにCroppedBitmapを取り直すため、
        /// トーンカーブ変更やAI処理完了などでMainImage.Sourceが更新された場合も内容が正しく追従する。
        /// 回転中は、切り出しビットマップ自体も画面表示と同じ角度だけ回転させる（ルーペ機能と同じ対応）。
        /// ドラッグ開始時・ページ再表示時など「内容が変わりうるタイミング」で使うやや重い版。
        /// </summary>
        private void SetupPageOffsetUnit(bool isLeft, Rect localRect, double fitScale, System.Windows.Point offsetPixels)
        {
            if (localRect.Width <= 0 || localRect.Height <= 0) return;
            if (MainImage.Source is not System.Windows.Media.Imaging.BitmapSource bmp) return;

            EnsurePageOffsetOverlayCanvas();

            var unit = isLeft
                ? (_leftOffsetUnit ??= CreatePageOffsetUnit())
                : (_rightOffsetUnit ??= CreatePageOffsetUnit());

            if (unit.Mask.Parent == null) _pageOffsetOverlayCanvas!.Children.Add(unit.Mask);
            if (unit.CropImage.Parent == null) _pageOffsetOverlayCanvas!.Children.Add(unit.CropImage);

            var maskScreen = LocalRectToScreen(localRect);
            unit.Mask.Width = maskScreen.Width;
            unit.Mask.Height = maskScreen.Height;
            System.Windows.Controls.Canvas.SetLeft(unit.Mask, maskScreen.X);
            System.Windows.Controls.Canvas.SetTop(unit.Mask, maskScreen.Y);

            int effectiveLeftWidthPx = (int)Math.Round(GetEffectiveSpreadLeftWidth(bmp));
            int cropX = isLeft ? 0 : effectiveLeftWidthPx;
            int cropW = isLeft ? effectiveLeftWidthPx : Math.Max(0, bmp.PixelWidth - effectiveLeftWidthPx);
            int cropH = bmp.PixelHeight;
            if (cropW <= 0 || cropH <= 0) return;

            var cropped = new System.Windows.Media.Imaging.CroppedBitmap(bmp, new Int32Rect(cropX, 0, cropW, cropH));
            unit.CropImage.Source = _rotationAngle == 0
                ? cropped
                : new System.Windows.Media.Imaging.TransformedBitmap(
                    cropped, new System.Windows.Media.RotateTransform(_rotationAngle * 90));

            var offsetLocalRect = new Rect(
                localRect.X + offsetPixels.X * fitScale,
                localRect.Y + offsetPixels.Y * fitScale,
                localRect.Width, localRect.Height);
            var cropScreen = LocalRectToScreen(offsetLocalRect);

            unit.CropImage.Width = cropScreen.Width;
            unit.CropImage.Height = cropScreen.Height;
            System.Windows.Controls.Canvas.SetLeft(unit.CropImage, cropScreen.X);
            System.Windows.Controls.Canvas.SetTop(unit.CropImage, cropScreen.Y);
        }

        /// <summary>
        /// 既にSetupPageOffsetUnitで内容（CroppedBitmap）が設定済みの前提で、位置のみを更新する軽量版。
        /// ドラッグ中のMouseMoveのたびに呼ばれるため、画像処理を一切行わずCanvas.Left/Topの更新のみに留める。
        /// </summary>
        private void MovePageOffsetUnit(bool isLeft, Rect localRect, double fitScale, System.Windows.Point offsetPixels)
        {
            var unit = isLeft ? _leftOffsetUnit : _rightOffsetUnit;
            if (unit == null) return;

            var offsetLocalRect = new Rect(
                localRect.X + offsetPixels.X * fitScale,
                localRect.Y + offsetPixels.Y * fitScale,
                localRect.Width, localRect.Height);
            var cropScreen = LocalRectToScreen(offsetLocalRect);

            unit.CropImage.Width = cropScreen.Width;
            unit.CropImage.Height = cropScreen.Height;
            System.Windows.Controls.Canvas.SetLeft(unit.CropImage, cropScreen.X);
            System.Windows.Controls.Canvas.SetTop(unit.CropImage, cropScreen.Y);
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
    }
}
