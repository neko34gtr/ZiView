using System;
using System.Globalization;
using System.Windows.Data;

namespace ZiView
{
    /// <summary>
    /// UIレイアウト中央座標導出用コンバーター
    /// </summary>
    public class HalfLayoutConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double doubleValue) return doubleValue / 2.0;
            return 0.0;
        }
        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) => throw new NotImplementedException();
    }
}
