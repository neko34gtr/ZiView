using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;

namespace ZiView
{
    /// <summary>
    /// SettingsWindow:
    /// システム設定（モデルフォルダ・推論エンジン）と、AIモデル管理（カテゴリ編集・モデル単位の割当/有効無効）
    /// をタブで分けて扱う設定ウィンドウ。「保存して閉じる」が押された場合のみAppConfigへ変更を反映する。
    /// </summary>
    public partial class SettingsWindow : Window
    {
        [DllImport("dwmapi.dll")]
        private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);
        private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

        public bool SettingsChanged { get; private set; } = false;

        // モデル行のカテゴリComboBoxが {Binding DataContext.Categories, RelativeSource={RelativeSource AncestorType=Window}} で参照する
        public ObservableCollection<string> Categories => _categories;

        private readonly AppConfig _config;
        private readonly ObservableCollection<ModelEntry> _entries = new();
        private readonly ObservableCollection<string> _categories = new();
        private string _selectedModelFolder;

        // モデル単位の行データ。カテゴリはComboBoxからのTwoWayバインド対象のため変更通知が必要。
        private class ModelEntry : INotifyPropertyChanged
        {
            public string FileName { get; set; } = "";

            private string _category = "";
            public string Category
            {
                get => _category;
                set { if (_category == value) return; _category = value; OnPropertyChanged(); }
            }

            public bool IsEnabled { get; set; }

            public event PropertyChangedEventHandler? PropertyChanged;
            private void OnPropertyChanged([CallerMemberName] string? name = null) =>
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }

        public SettingsWindow(AppConfig config)
        {
            InitializeComponent();

            // メインウィンドウと同じダークタイトルバーを適用（OS標準の白いタイトルバーを防止）
            this.SourceInitialized += (s, e) =>
            {
                IntPtr hwnd = new WindowInteropHelper(this).Handle;
                int darkMode = 1;
                DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ref darkMode, sizeof(int));
            };

            this.DataContext = this;

            _config = config;
            _selectedModelFolder = _config.ModelFolder;

            RadioTensorRt.IsChecked = _config.EnginePreference != "CUDA";
            RadioCuda.IsChecked = _config.EnginePreference == "CUDA";
            TrtCacheCheckBox.IsChecked = _config.TensorRtEngineCacheEnabled;
            UpdateTrtCacheSizeText();

            ModelFolderTextBox.Text = MainWindow.GetModelDirectory(_selectedModelFolder);

            CategoryListControl.ItemsSource = _categories;
            ModelListControl.ItemsSource = _entries;

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

        /// <summary>
        /// モデルフォルダをスキャンし、カテゴリ一覧・モデル一覧を再構築する。
        /// カテゴリは「モデルに実際に割り当てられているもの」＋「保存済みのカスタムカテゴリ（空でも保持）」の和集合。
        /// </summary>
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
                            Category = MainWindow.GetEffectiveModelCategory(_config, f),
                            IsEnabled = !_config.ExcludedModels.Contains(f)
                        });
                    }
                }

                _categories.Clear();
                var allCategories = _entries.Select(x => x.Category)
                    .Concat(_config.CustomCategories)
                    .Where(c => !string.IsNullOrWhiteSpace(c))
                    .Distinct()
                    .OrderBy(c => c == "その他" ? 1 : 0)
                    .ThenBy(c => c);
                foreach (var c in allCategories) _categories.Add(c);

                EmptyListText.Visibility = _entries.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
            }
            catch { EmptyListText.Visibility = Visibility.Visible; }
        }

        private void AddCategoryButton_Click(object sender, RoutedEventArgs e)
        {
            string name = NewCategoryTextBox.Text?.Trim() ?? "";
            if (string.IsNullOrEmpty(name)) return;

            if (!_categories.Contains(name)) _categories.Add(name);
            NewCategoryTextBox.Text = "";
        }

        private void DeleteCategoryButton_Click(object sender, RoutedEventArgs e)
        {
            if (sender is not FrameworkElement fe || fe.DataContext is not string category) return;

            // このカテゴリを使用しているモデルは既定の自動分類へ戻す
            foreach (var entry in _entries.Where(x => x.Category == category))
            {
                entry.Category = MainWindow.GetModelCategory(entry.FileName);
                if (!_categories.Contains(entry.Category)) _categories.Add(entry.Category);
            }

            _categories.Remove(category);
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

            // 既定の自動分類と異なるものだけを差分として保存する
            _config.ModelCategoryOverrides = _entries
                .Where(x => x.Category != MainWindow.GetModelCategory(x.FileName))
                .ToDictionary(x => x.FileName, x => x.Category);

            // モデルが1つも割り当てられていないカテゴリも消えないよう保持する
            _config.CustomCategories = _categories
                .Where(c => !_entries.Any(x => x.Category == c))
                .ToList();

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
