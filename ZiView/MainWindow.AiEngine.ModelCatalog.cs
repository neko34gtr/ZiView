using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: AIモデル（*.onnx）の検出・カテゴリ分類・選択UI（Category/ModelComboBox）、
    /// およびモデル切替時のセッション再構築を担当する。
    /// </summary>
    public partial class MainWindow
    {
        private Dictionary<string, List<string>> _modelsByCategory = new();

        private bool _isUpdatingModelComboInternal = false;

        // 既知モデルのファイル名 → 特性・用途カテゴリの対応表。
        // 未知の *.onnx が追加された場合は GetModelCategory 内のキーワード推定でフォールバックする。
        private static readonly Dictionary<string, string> _modelCategoryMap = new(StringComparer.OrdinalIgnoreCase)
        {
            { "RealESRGAN_x4plus_anime_6B.onnx", "アニメ・コミック" },
            { "4x_cugan_pretrain.onnx", "漫画・イラスト（線画）" },
            { "4x-UltraSharpV2_fp32_op17.onnx", "汎用（実写/イラスト）" },
            { "4xRealWebPhoto_v4_drct-l_fp32.onnx", "実写特化" },
            { "4xNomos2_realplksr_dysample_256_fp32_fullyoptimized.onnx", "漫画・アニメ（高精細）" },
            { "1xDenoise_realplksr_otf_fp32.onnx", "ノイズ除去" },
        };

        /// <summary>
        /// ユーザーが設定ウィンドウで手動割り当てしたカテゴリ（あれば）を優先し、
        /// 無ければ既定の自動分類(GetModelCategory)にフォールバックする。
        /// </summary>
        internal static string GetEffectiveModelCategory(AppConfig config, string fileName)
        {
            if (config.ModelCategoryOverrides.TryGetValue(fileName, out var overridden) && !string.IsNullOrWhiteSpace(overridden))
                return overridden;
            return GetModelCategory(fileName);
        }

        /// <summary>
        /// 設定された ModelFolder を絶対パスへ解決する。
        /// 空文字はプログラムルート（従来互換）、相対パスはルートからの相対、絶対パスはそのまま使用する。
        /// </summary>
        internal static string GetModelDirectory(string configuredFolder)
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            if (string.IsNullOrWhiteSpace(configuredFolder)) return root;
            return Path.IsPathRooted(configuredFolder) ? configuredFolder : Path.Combine(root, configuredFolder);
        }

        internal static string GetModelCategory(string fileName)
        {
            if (_modelCategoryMap.TryGetValue(fileName, out var cat)) return cat;

            // 未知のモデル向けフォールバック（ファイル名からの推定。確実な分類ではない）
            string lower = fileName.ToLowerInvariant();
            if (lower.Contains("denoise")) return "ノイズ除去";
            if (lower.Contains("cugan")) return "漫画・イラスト（線画）";
            if (lower.Contains("webphoto") || lower.Contains("photo")) return "実写特化";
            if (lower.Contains("nomos")) return "漫画・アニメ（高精細）";
            if (lower.Contains("anime") || lower.Contains("comic")) return "アニメ・コミック";
            if (lower.Contains("sharp") || lower.Contains("general")) return "汎用（実写/イラスト）";
            return "その他";
        }

        /// <summary>
        /// Transformer/Attention系アーキテクチャ（DRCT等）はタイルサイズが大きいとVRAM消費・処理時間が
        /// 急増しやすい傾向があるため、既知の重量級モデルは既定タイルサイズを予防的に下げる。
        /// （未検証の予防的措置であり、確実な不具合修正ではない）
        /// </summary>
        private static int GetTileSizeForModel(string fileName)
        {
            string lower = fileName.ToLowerInvariant();
            if (lower.Contains("drct") || lower.Contains("nomos2")) return 192;
            return 256;
        }

        /// <summary>
        /// プログラムルート直下の *.onnx ファイルを列挙し、ModelComboBoxへ反映する。
        /// 設定に保存済みのモデル名が存在すればそれを、無ければ先頭のモデルを選択状態にする。
        /// 除外リスト(_config.ExcludedModels)に含まれるモデルは一覧から除かれる。
        /// </summary>
        private void ScanOnnxModels()
        {
            try
            {
                string root = GetModelDirectory(_config.ModelFolder);
                if (!Directory.Exists(root))
                {
                    try
                    {
                        Directory.CreateDirectory(root);
                        WriteLog($"Model folder did not exist, created: {root}");
                    }
                    catch (Exception exCreate)
                    {
                        WriteLog($"Model folder creation failed: {exCreate.Message}");
                    }
                }

                var files = Directory.Exists(root)
                    ? Directory.GetFiles(root, "*.onnx")
                        .Select(Path.GetFileName)
                        .Where(f => !string.IsNullOrEmpty(f))
                        .Select(f => f!)
                        .Where(f => !_config.ExcludedModels.Contains(f))
                        .OrderBy(f => f)
                        .ToList()
                    : new List<string>();

                if (files.Count == 0)
                {
                    WriteLog($"ERROR: No usable .onnx model files found in '{root}' (all excluded or missing).");
                    StatusText.Text = "Mode: Model Missing";
                    CategoryComboBox.ItemsSource = null;
                    ModelComboBox.ItemsSource = null;
                    return;
                }

                _modelsByCategory = files
                    .GroupBy(f => GetEffectiveModelCategory(_config, f))
                    .OrderBy(g => g.Key == "その他" ? 1 : 0)
                    .ThenBy(g => g.Key)
                    .ToDictionary(g => g.Key, g => g.OrderBy(x => x).ToList());

                // 保存済みモデルが属するカテゴリを特定し、無ければ先頭カテゴリを初期選択
                string targetCategory = _modelsByCategory
                    .FirstOrDefault(kv => kv.Value.Contains(_config.SelectedModel)).Key
                    ?? _modelsByCategory.Keys.First();

                string targetModel = _modelsByCategory[targetCategory].Contains(_config.SelectedModel)
                    ? _config.SelectedModel
                    : _modelsByCategory[targetCategory][0];

                _isUpdatingModelComboInternal = true;
                CategoryComboBox.ItemsSource = _modelsByCategory.Keys.ToList();
                CategoryComboBox.SelectedItem = targetCategory;
                ModelComboBox.ItemsSource = _modelsByCategory[targetCategory];
                ModelComboBox.SelectedItem = targetModel;
                _isUpdatingModelComboInternal = false;

                _config.SelectedModel = targetModel;
            }
            catch (Exception ex) { WriteLog($"Model Scan Error: {ex.Message}"); }
        }

        private void CategoryComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (_isUpdatingModelComboInternal) return;
            if (CategoryComboBox.SelectedItem is not string category) return;
            if (!_modelsByCategory.TryGetValue(category, out var models) || models.Count == 0) return;

            // カテゴリ変更時は、そのカテゴリ内の先頭モデルを自動選択する
            _isUpdatingModelComboInternal = true;
            ModelComboBox.ItemsSource = models;
            ModelComboBox.SelectedItem = models[0];
            _isUpdatingModelComboInternal = false;

            _ = ApplyModelSelectionAsync(models[0]);
        }

        private async void ModelComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (_isUpdatingModelComboInternal) return;
            if (ModelComboBox.SelectedItem is not string modelFileName) return;
            await ApplyModelSelectionAsync(modelFileName);
        }

        private async Task ApplyModelSelectionAsync(string modelFileName)
        {
            if (modelFileName == _config.SelectedModel && _onnxSession != null) return;

            // 実行中の推論を止め、_onnxSessionへのアクセスが完全に終わるまで待つ
            // （待機自体はawaitのためUIスレッドはブロックされない）
            _cts?.Cancel();
            ClearPageCache(); // 先読みキャッシュはモデル切替前のAI推論結果を保持しているため無効化する
            var runningTask = _currentInferenceTask;
            if (runningTask != null)
            {
                try { await runningTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }

            _config.SelectedModel = modelFileName;

            _onnxSession?.Dispose();
            _onnxSession = null;
            _inputName = null;

            StatusText.Text = "Mode: Loading model...";
            await InitializeAiWithOverlayAsync(modelFileName);

            // モデル切替後、表示中ページをAI再変換
            if (_imageList.Count > 0)
            {
                RefreshDisplay();
            }
        }
    }
}
