using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace ZiView
{
    /// <summary>
    /// CustomColorPicker:
    /// HSVモデル（色相・彩度・明度）を用いて直感的にカラーコードを生成するカスタムカラーピッカーコンポーネントです。
    /// </summary>
    public class CustomColorPicker
    {
        private readonly Canvas _svCanvas;
        private readonly Slider _hueSlider;
        private readonly Rectangle _previewBox;
        private readonly Action<string> _onColorApplied;

        private double _currentHue = 0;
        private double _currentSaturation = 1.0;
        private double _currentValue = 1.0;

        public CustomColorPicker(Canvas svCanvas, Slider hueSlider, Rectangle previewBox, Action<string> onColorApplied)
        {
            _svCanvas = svCanvas;
            _hueSlider = hueSlider;
            _previewBox = previewBox;
            _onColorApplied = onColorApplied;

            // イベントの紐付け
            _svCanvas.MouseLeftButtonDown += OnSvCanvas_MouseDown;
            _svCanvas.MouseMove += OnSvCanvas_MouseMove;
            _hueSlider.ValueChanged += OnHueSlider_ValueChanged;

            // 初期色相設定の同期
            _currentHue = _hueSlider.Value;
        }

        private void OnHueSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            _currentHue = e.NewValue;
            UpdatePreview();
        }

        private void OnSvCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            ProcessSvMouse(e);
        }

        private void OnSvCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                ProcessSvMouse(e);
            }
        }

        private void ProcessSvMouse(MouseEventArgs e)
        {
            var pos = e.GetPosition(_svCanvas);
            _currentSaturation = Math.Max(0.0, Math.Min(1.0, pos.X / _svCanvas.ActualWidth));
            _currentValue = Math.Max(0.0, Math.Min(1.0, 1.0 - (pos.Y / _svCanvas.ActualHeight)));

            UpdatePreview();
        }

        private void UpdatePreview()
        {
            string hex = HsvToHex(_currentHue, _currentSaturation, _currentValue);
            try
            {
                var obj = new BrushConverter().ConvertFromString(hex);
                if (obj is SolidColorBrush brush)
                {
                    _previewBox.Fill = brush;
                }
            }
            catch { }
        }

        public void ApplyColor()
        {
            string hex = HsvToHex(_currentHue, _currentSaturation, _currentValue);
            _onColorApplied?.Invoke(hex);
        }

        private string HsvToHex(double hue, double saturation, double value)
        {
            int hi = (int)(Math.Floor(hue / 60.0) % 6);
            double f = hue / 60.0 - Math.Floor(hue / 60.0);

            value = value * 255.0;
            byte v = (byte)value;
            byte p = (byte)(value * (1.0 - saturation));
            byte q = (byte)(value * (1.0 - f * saturation));
            byte t = (byte)(value * (1.0 - (1.0 - f) * saturation));

            byte r = 0, g = 0, b = 0;

            switch (hi)
            {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                case 5: r = v; g = p; b = q; break;
            }

            return $"#{r:X2}{g:X2}{b:X2}";
        }
    }
}