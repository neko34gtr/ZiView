using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: カスタム背景色ピッカー（HSVモデルでの色相・彩度・明度演算とプレビュー描画）を担当する。
    /// </summary>
    public partial class MainWindow
    {
        // カラーピッカー用状態変数
        private double _currentHue = 0.0;
        private double _currentSaturation = 1.0;
        private double _currentValue = 1.0;
        private string _selectedHexColor = "#000000";

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
    }
}
