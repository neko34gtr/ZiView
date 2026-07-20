using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;

namespace ZiView
{
    /// <summary>
    /// SettingsWindow:
    /// AIモデルの有効/無効管理、および推論エンジンの優先モード切替を行う専用設定ウィンドウ。
    /// 「保存して閉じる」が押された場合のみ、渡されたAppConfigへ変更を反映する。
    /// </summary>
    public partial class SettingsWindow : Window
    {
        public bool SettingsChanged { get; private set; } = false;

        private readonly AppConfig _config;
        private readonly List<ModelEntry> _entries = new();
        private string _selectedModelFolder;

        private class ModelEntry
        {
            public string FileName { get; set; } = "";
            public string Category { get; set; } = "";
            public bool IsEnabled { get; set; }
        }

        public SettingsWindow(AppConfig config)
        {
            InitializeComponent();
            _config = config;
            _selectedModelFolder = _config.ModelFolder;

            RadioTensorRt.IsChecked = _config.EnginePreference != "CUDA";
            RadioCuda.IsChecked = _config.EnginePreference == "CUDA";
            TrtCacheCheckBox.IsChecked = _config.TensorRtEngineCacheEnabled;
            UpdateTrtCacheSizeText();

            ModelFolderTextBox.Text = MainWindow.GetModelDirectory(_selectedModelFolder);
            LoadModelList();
        }

        private void UpdateTrtCacheSizeText()
        {
            try
            {
                string cacheDir = MainWindow.GetTensorRtCacheDirectory();
                if (!Directory.Exists(cacheDir))
                {
                    TrtCacheSizeText.Text = "キャッシュサイズ: 0 MB（未生成）";
                    return;
                }
                long totalBytes = new DirectoryInfo(cacheDir).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
                TrtCacheSizeText.Text = $"キャッシュサイズ: {totalBytes / 1024.0 / 1024.0:F1} MB";
            }
            catch
            {
                TrtCacheSizeText.Text = "キャッシュサイズ: 取得失敗";
            }
        }

        private void ClearTrtCacheButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string cacheDir = MainWindow.GetTensorRtCacheDirectory();
                if (Directory.Exists(cacheDir))
                {
                    var result = MessageBox.Show(
                        "TensorRTエンジンキャッシュを削除します。\n次回TensorRT利用時、初回相当のビルド時間がかかり直します。\n\nよろしいですか？",
                        "キャッシュの削除", MessageBoxButton.YesNo, MessageBoxImage.Question);
                    if (result != MessageBoxResult.Yes) return;

                    Directory.Delete(cacheDir, true);
                }
                UpdateTrtCacheSizeText();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"キャッシュの削除に失敗しました: {ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void BrowseModelFolder_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "AIモデルフォルダを選択",
                InitialDirectory = MainWindow.GetModelDirectory(_selectedModelFolder)
            };

            if (dialog.ShowDialog(this) == true)
            {
                _selectedModelFolder = dialog.FolderName;
                ModelFolderTextBox.Text = _selectedModelFolder;
                LoadModelList();
            }
        }

        private void LoadModelList()
        {
            try
            {
                string root = MainWindow.GetModelDirectory(_selectedModelFolder);
                _entries.Clear();

                if (Directory.Exists(root))
                {
                    var files = Directory.GetFiles(root, "*.onnx")
                        .Select(Path.GetFileName)
                        .Where(f => !string.IsNullOrEmpty(f))
                        .Select(f => f!)
                        .OrderBy(f => f)
                        .ToList();

                    foreach (var f in files)
                    {
                        _entries.Add(new ModelEntry
                        {
                            FileName = f,
                            Category = MainWindow.GetModelCategory(f),
                            IsEnabled = !_config.ExcludedModels.Contains(f)
                        });
                    }
                }

                ModelListControl.ItemsSource = null;
                ModelListControl.ItemsSource = _entries;
                EmptyListText.Visibility = _entries.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
            }
            catch { EmptyListText.Visibility = Visibility.Visible; }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            // 少なくとも1つは有効なモデルを残す（全除外での起動不能を防ぐ）
            if (_entries.Count > 0 && _entries.All(x => !x.IsEnabled))
            {
                MessageBox.Show("すべてのモデルを無効にすることはできません。少なくとも1つは有効にしてください。",
                    "設定", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            _config.ExcludedModels = _entries.Where(x => !x.IsEnabled).Select(x => x.FileName).ToList();
            _config.EnginePreference = (RadioCuda.IsChecked == true) ? "CUDA" : "TensorRT";
            _config.TensorRtEngineCacheEnabled = TrtCacheCheckBox.IsChecked ?? true;
            _config.ModelFolder = _selectedModelFolder;

            SettingsChanged = true;
            DialogResult = true;
            Close();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
