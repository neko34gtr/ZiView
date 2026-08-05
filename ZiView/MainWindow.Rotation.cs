namespace ZiView
{
    /// <summary>
    /// MainWindow partial: 表示回転機能（Rキー）を担当する。
    /// MainImage.LayoutTransformへRotateTransformを適用し、Stretch=Uniformと組み合わせて
    /// 90/270度回転時も自動的にウィンドウへ収まるサイズへフィットさせる。設定として永続化はしない。
    /// </summary>
    public partial class MainWindow
    {
        // 表示回転（Rキー）: 設定として永続化はせず、アプリ再起動または別ソース読込でリセットされる
        private System.Windows.Media.RotateTransform? _imgRotate;
        private int _rotationAngle = 0; // 0-3（90度単位）

        /// <summary>
        /// MainImage.LayoutTransformへ回転専用のRotateTransformを1度だけ組み込む。
        /// ズーム・パン用のImgScale/ImgTranslateはRenderTransform（レイアウト確定後に描画だけをずらす）のままにし、
        /// 回転はあえてLayoutTransform側に持たせる。LayoutTransformは測定・配置(レイアウト)の段階で効くため、
        /// 90/270度回転で縦横が入れ替わった場合でもWPFが自動的にMainImageのレイアウト枠を再計算し、
        /// Stretch=Uniformと合わせてウィンドウ（コンテナ）に収まるサイズへ自動フィットする。
        /// RenderTransformのままだと枠は元の縦横のまま変わらず、回転後の画像だけがはみ出す/隙間ができる問題があったための対応。
        /// </summary>
        private void EnsureRotateTransform()
        {
            if (_imgRotate != null) return;
            _imgRotate = new System.Windows.Media.RotateTransform(0);
            MainImage.LayoutTransform = _imgRotate;
        }

        /// <summary>
        /// Rキー/コンテキストメニューから呼ばれる表示回転。押すたびに右90度ずつ進み、4回で元に戻る。
        /// 見開き表示中（L2/R2）に呼ばれた場合は単ページ(S1)へ切り替えてから回転する
        /// （基準は現在のPageSlider.Valueのページ）。開き方向の切替自体はセッション内一時状態のため、
        /// 再起動すれば元の設定に戻る。回転自体も設定として永続化しない（アプリ再起動でリセット、
        /// 別ソース読込でもResetTransformによりリセットされる）。
        /// </summary>
        private void RotateView()
        {
            if (_readingMode != PageOpenMode.Single)
            {
                SetReadingMode(PageOpenMode.Single);
            }

            EnsureRotateTransform();
            _rotationAngle = (_rotationAngle + 1) % 4;
            _imgRotate!.Angle = _rotationAngle * 90;
            RefreshPageOffsetOverlays();

            ShowNotification($"回転: {_rotationAngle * 90}°");
        }
    }
}
