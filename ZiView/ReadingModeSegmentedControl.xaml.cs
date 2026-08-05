using System;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using static System.Net.Mime.MediaTypeNames;

namespace ZiView
{
    /// <summary>
    /// L2（左開き）/ S1（単ページ）/ R2（右開き）を切り替えるカプセル型セグメントコントロール。
    /// クリックで選択状態が切り替わり、ModeChangedイベントで通知する。表示のみを担当し、
    /// 実際のページ合成ロジックへの反映はMainWindow.ReadingMode.cs側（ModeChangedの購読）が行う。
    /// </summary>
    public partial class ReadingModeSegmentedControl : UserControl
    {
        private static readonly SolidColorBrush NormalBrush = CreateFrozenBrush(0x3B, 0x3B, 0x3B);
        private static readonly SolidColorBrush HoverBrush = CreateFrozenBrush(0x4F, 0x4F, 0x4F);
        private static readonly SolidColorBrush SelectedBrush = CreateFrozenBrush(0x3B, 0x7D, 0xD8);
        private static readonly SolidColorBrush NormalTextBrush = CreateFrozenBrush(0xDD, 0xDD, 0xDD);
        private static readonly SolidColorBrush SelectedTextBrush = CreateFrozenBrush(0xFF, 0xFF, 0xFF);

        private static SolidColorBrush CreateFrozenBrush(byte r, byte g, byte b)
        {
            var brush = new SolidColorBrush(Color.FromRgb(r, g, b));
            brush.Freeze();
            return brush;
        }

        /// <summary>選択モードが変わるたびに発火する。</summary>
        public event EventHandler<PageOpenMode>? ModeChanged;

        private PageOpenMode _mode = PageOpenMode.RightOpen;

        /// <summary>現在選択中のモード。コード側から代入すると表示だけが更新され、ModeChangedは発火しない。</summary>
        public PageOpenMode Mode
        {
            get => _mode;
            set
            {
                if (_mode == value) return;
                _mode = value;
                UpdateVisualState();
            }
        }

        public ReadingModeSegmentedControl()
        {
            InitializeComponent();
            Loaded += (s, e) => UpdateVisualState();
        }

        private Border GetSegmentBorder(PageOpenMode mode) => mode switch
        {
            PageOpenMode.LeftOpen => SegL2,
            PageOpenMode.Single => SegS1,
            _ => SegR2,
        };

        private TextBlock GetSegmentText(PageOpenMode mode) => mode switch
        {
            PageOpenMode.LeftOpen => TextL2,
            PageOpenMode.Single => TextS1,
            _ => TextR2,
        };

        private void UpdateVisualState()
        {
            foreach (PageOpenMode mode in new[] { PageOpenMode.LeftOpen, PageOpenMode.Single, PageOpenMode.RightOpen })
            {
                bool selected = mode == _mode;
                GetSegmentBorder(mode).Background = selected ? SelectedBrush : NormalBrush;
                GetSegmentText(mode).Foreground = selected ? SelectedTextBrush : NormalTextBrush;
            }
        }

        private static bool TryGetModeFromSender(object sender, out PageOpenMode mode)
        {
            mode = default;
            if (sender is not Border border || border.Tag is not string tagStr) return false;
            return Enum.TryParse(tagStr, out mode);
        }

        private void Segment_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (!TryGetModeFromSender(sender, out var mode)) return;
            if (mode == _mode) return;

            _mode = mode;
            UpdateVisualState();
            ModeChanged?.Invoke(this, _mode);
        }

        private void Segment_MouseEnter(object sender, MouseEventArgs e)
        {
            if (sender is not Border border) return;
            if (TryGetModeFromSender(sender, out var mode) && mode == _mode) return; // 選択中はホバー色で上書きしない
            border.Background = HoverBrush;
        }

        private void Segment_MouseLeave(object sender, MouseEventArgs e)
        {
            if (sender is not Border border) return;
            if (TryGetModeFromSender(sender, out var mode) && mode == _mode) return;
            border.Background = NormalBrush;
        }
    }
}
