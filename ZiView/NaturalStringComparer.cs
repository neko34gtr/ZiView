using System.Collections.Generic;

namespace ZiView
{
    /// <summary>
    /// Windowsエクスプローラー(StrCmpLogicalW)相当の自然順ソートを行うComparer。
    /// 半角数字だけでなく全角数字（０-９）も数値として認識するため、
    /// 「画像１」「画像１０」のような桁数不揃いの全角連番でも正しい順序になる。
    /// </summary>
    public class NaturalStringComparer : IComparer<string>
    {
        public static readonly NaturalStringComparer Instance = new();

        public int Compare(string? x, string? y)
        {
            if (x == null || y == null) return string.CompareOrdinal(x, y);

            int ix = 0, iy = 0;
            while (ix < x.Length && iy < y.Length)
            {
                char cx = x[ix], cy = y[iy];

                if (IsDigit(cx) && IsDigit(cy))
                {
                    int sx = ix, sy = iy;
                    while (ix < x.Length && IsDigit(x[ix])) ix++;
                    while (iy < y.Length && IsDigit(y[iy])) iy++;

                    string numX = NormalizeDigits(x.Substring(sx, ix - sx));
                    string numY = NormalizeDigits(y.Substring(sy, iy - sy));

                    // 先頭ゼロを除いた桁数・値で数値として比較する
                    string trimX = numX.TrimStart('0');
                    string trimY = numY.TrimStart('0');
                    if (trimX.Length == 0) trimX = "0";
                    if (trimY.Length == 0) trimY = "0";

                    int cmp = trimX.Length != trimY.Length
                        ? trimX.Length.CompareTo(trimY.Length)
                        : string.CompareOrdinal(trimX, trimY);
                    if (cmp != 0) return cmp;

                    // 数値としては同値（例: "01" と "1"）の場合は、元の表記で決着させる
                    cmp = string.CompareOrdinal(numX, numY);
                    if (cmp != 0) return cmp;
                }
                else
                {
                    int cmp = char.ToUpperInvariant(cx).CompareTo(char.ToUpperInvariant(cy));
                    if (cmp != 0) return cmp;
                    ix++; iy++;
                }
            }

            return (x.Length - ix).CompareTo(y.Length - iy);
        }

        private static bool IsDigit(char c) =>
            (c >= '0' && c <= '9') || (c >= '\uFF10' && c <= '\uFF19');

        // 全角数字０-９を半角0-9へ正規化する（比較用。表記自体は変更しない）
        private static string NormalizeDigits(string s)
        {
            var chars = new char[s.Length];
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                chars[i] = (c >= '\uFF10' && c <= '\uFF19') ? (char)(c - '\uFF10' + '0') : c;
            }
            return new string(chars);
        }
    }
}
