namespace ZiView
{
    /// <summary>
    /// MainWindow partial: 見開きの開き方向切替（L2/S1/R2セグメントコントロール）を担当する。
    /// 従来の見開きON/OFFのみだったCheckSpread（チェックボックス）を置き換え、
    /// 右開き（従来のマンガ方式）・左開き（洋書方式）・単ページの3状態を切り替えられるようにする。
    ///
    /// 【XAML側で必要な作業】MainWindow.xaml内の既存の見開きチェックボックス（CheckSpread）を、
    /// 本パーシャルに対応する<local:ReadingModeSegmentedControl x:Name="ReadingModeControl"
    /// ModeChanged="ReadingModeControl_ModeChanged"/>へ置き換えること（xmlns:localはこのプロジェクトの
    /// クラス名前空間を指すよう指定）。置き換え後、CheckSpread.IsChecked/Checked/Uncheckedへの参照は
    /// コード側に一切残っていない（本セッションで全箇所をPageOpenMode/_readingModeベースへ置換済み）。
    /// </summary>
    public partial class MainWindow
    {
        // 既定は従来通りの右開き（マンガ方式）。config.jsonへは保存しない（回転・トーンカーブ・ルーペと同様、
        // セッション内一時状態として扱う。永続化したい場合はAppConfigへ項目追加のうえSetReadingMode/
        // 起動時のApplyConfigToUi相当の箇所で読み書きすること）
        private PageOpenMode _readingMode = PageOpenMode.RightOpen;

        /// <summary>ReadingModeSegmentedControlのModeChangedイベントハンドラ（XAML側で配線）。</summary>
        private void ReadingModeControl_ModeChanged(object? sender, PageOpenMode mode)
        {
            SetReadingMode(mode);
        }

        /// <summary>
        /// 見開きの開き方向を切り替える。UIコントロールの選択表示も同期し、
        /// 既存のOnSettingChanged（見開き設定変更時の処理）と同じくキャッシュ破棄＋再表示を行う。
        /// RotateView等、コード側から直接切り替えたい場合もこのメソッドを呼ぶこと。
        /// </summary>
        private void SetReadingMode(PageOpenMode mode)
        {
            if (_readingMode == mode) return;
            _readingMode = mode;

            if (ReadingModeControl != null)
            {
                ReadingModeControl.Mode = mode;
            }

            ClearPageCache();
            RefreshDisplay();
        }
    }
}
