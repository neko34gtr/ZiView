using System.Collections.Generic;

namespace ZiView
{
    /// <summary>
    /// AppConfigクラス:
    /// アプリケーションの永続的な設定（ウィンドウ位置、チェックボックスの状態、最後に開いたパス等）を管理します。
    /// </summary>
    public class AppConfig
    {
        public double WindowLeft { get; set; } = 100;
        public double WindowTop { get; set; } = 100;
        public double WindowWidth { get; set; } = 1200;
        public double WindowHeight { get; set; } = 800;
        public bool CheckSpread { get; set; } = false;
        public bool CheckAutoDetectOrder { get; set; } = false; // プロパティ名修正等の不整合防止
        public bool CheckAutoDetect { get; set; } = false;
        public bool CheckPrefetch { get; set; } = false;
        public double SplitSliderValue { get; set; } = 100;
        public string LastSourcePath { get; set; } = string.Empty;

        // レンズ補正およびレティクル用永続パラメータ
        public bool ShowReticle { get; set; } = true;
        public bool EnableLensCorrection { get; set; } = false;
        public double LensCorrectionAmount { get; set; } = 0.40;

        // 背景色の設定を保存するプロパティ（デフォルト: 真の黒）
        public string BackgroundColor { get; set; } = "#000000";

        // AI推論の有効/無効（リアルタイム切替。OFF時は原寸画像をそのまま表示する）
        public bool EnableAiInference { get; set; } = true;

        // 選択中のAIモデルファイル名（プログラムルート直下の *.onnx）
        public string SelectedModel { get; set; } = "RealESRGAN_x4plus_anime_6B.onnx";

        // 動作が重すぎる等の理由でユーザーが選択肢から除外したモデル（ファイル名一覧）
        public List<string> ExcludedModels { get; set; } = new();

        // AI進捗OSDの表示位置: TopCenter / TopLeft / TopRight / BottomCenter
        public string AiOsdPosition { get; set; } = "TopCenter";

        // 推論エンジンの優先モード: "TensorRT"(TensorRT→CUDA→CPU) / "CUDA"(CUDA→CPUのみ)
        public string EnginePreference { get; set; } = "TensorRT";

        // AIモデル(*.onnx)を探すフォルダ。空文字はプログラムルート（従来互換）。
        // 相対パスの場合はプログラムルートからの相対、絶対パスならそのまま使用する。
        // 既定値はプログラムルート直下の .\model フォルダ。
        public string ModelFolder { get; set; } = @".\model";

        // ユーザーが設定ウィンドウで手動割り当てしたモデルのカテゴリ（ファイル名 → カテゴリ名）。
        // 既定の自動分類（GetModelCategory）と異なる場合のみ差分として保存される。
        public Dictionary<string, string> ModelCategoryOverrides { get; set; } = new();

        // ユーザーが作成したカスタムカテゴリ名（現在モデルが1つも割り当てられていないものも保持するため）。
        public List<string> CustomCategories { get; set; } = new();

        // TensorRTのエンジンキャッシュ(ビルド済み最適化エンジン)をディスクへ永続化するか。
        // ONにすると初回のみ低速（エンジンビルド）、2回目以降は大幅に高速化される。
        // ディスク容量が気になる場合はOFFにするか、SettingsWindowからキャッシュを削除できる。
        public bool TensorRtEngineCacheEnabled { get; set; } = true;
    }
}
