using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

using OpenCvSharp;
using OpenCvSharp.WpfExtensions;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: 画像/アーカイブソースの読み込み、ページ送り、
    /// 見開き合成、AI比較表示（分割スライダー）を担当する。
    /// </summary>
    public partial class MainWindow
    {
        private string? _currentSourcePath;
        private List<string> _imageList = new List<string>();
        private Mat? _currentCombinedOriginal;
        private Mat? _currentCombinedUpscaled;
        private readonly string _tempExtractDir = Path.Combine(Path.GetTempPath(), "ZiView_Temp");

        // 無限ループイベントを抑止するためのフラグ
        private bool _isUpdatingPageSliderInternal = false;

        public void LoadSource(string path)
        {
            WriteLog($"Loading source tree: {path}");
            _currentSourcePath = path;
            _imageList.Clear();
            int initialIndex = 0;
            try
            {
                if (Directory.Exists(path))
                {
                    var files = Directory.GetFiles(path)
                        .Where(f => !string.IsNullOrEmpty(f) && IsImageFile(f))
                        .OrderBy(f => f);
                    _imageList.AddRange(files);
                }
                else if (File.Exists(path))
                {
                    // パス自体が画像ファイルの場合は、親フォルダ内の画像を全てリスト化し、
                    // 自身の位置を初期表示ページとすることで、フォルダ指定時と同様に前後移動を可能にする
                    if (IsImageFile(path))
                    {
                        string? parentDir = Path.GetDirectoryName(path);
                        if (!string.IsNullOrEmpty(parentDir) && Directory.Exists(parentDir))
                        {
                            var files = Directory.GetFiles(parentDir)
                                .Where(f => !string.IsNullOrEmpty(f) && IsImageFile(f))
                                .OrderBy(f => f);
                            _imageList.AddRange(files);

                            int idx = _imageList.FindIndex(f => string.Equals(f, path, StringComparison.OrdinalIgnoreCase));
                            initialIndex = idx >= 0 ? idx : 0;
                        }
                        else
                        {
                            _imageList.Add(path);
                        }
                    }
                    else
                    {
                        string ext = Path.GetExtension(path).ToLower(CultureInfo.InvariantCulture);
                        if (ext == ".zip")
                        {
                            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                            using (ZipArchive archive = new ZipArchive(fs, ZipArchiveMode.Read))
                            {
                                var keys = archive.Entries
                                    .Where(e => !string.IsNullOrEmpty(e.FullName) && IsImageFile(e.FullName))
                                    .Select(e => e.FullName)
                                    .OrderBy(k => k);
                                _imageList.AddRange(keys);
                            }
                        }
                        else if (ext == ".rar" || ext == ".7z")
                        {
                            _imageList.AddRange(GetArchiveFileListViaCli(path));
                        }
                    }
                }
            }
            catch (Exception ex) { WriteLog($"Source Parser Error: {ex.Message}"); }

            if (_imageList.Count > 0)
            {
                if (initialIndex < 0 || initialIndex >= _imageList.Count) initialIndex = 0;

                _isUpdatingPageSliderInternal = true;
                PageSlider.Maximum = _imageList.Count - 1;
                PageSlider.Value = initialIndex;
                _isUpdatingPageSliderInternal = false;

                ResetTransform();
                DisplayPage(initialIndex);
            }
        }

        public void LoadImage(string path)
        {
            LoadSource(path);
        }

        private bool IsImageFile(string? f) =>
            !string.IsNullOrEmpty(f) && (
            f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase));

        private void RefreshDisplay() => DisplayPage((int)PageSlider.Value);

        private async void DisplayPage(int index)
        {
            _cts?.Cancel();
            var priorTask = _currentInferenceTask;
            if (priorTask != null)
            {
                // 前ページの推論が_currentCombinedOriginal等を参照中の可能性があるため、
                // 完全に終わってから破棄する（ObjectDisposedException対策）
                try { await priorTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }

            _cts = new System.Threading.CancellationTokenSource();
            var token = _cts.Token;

            if (index < 0 || index >= _imageList.Count) return;

            try
            {
                _currentCombinedOriginal?.Dispose();
                _currentCombinedUpscaled?.Dispose();
                _currentCombinedUpscaled = null;

                Mat pRight = LoadMat(_imageList[index]);
                bool isAutoSingle = (CheckAutoDetect.IsChecked == true && (double)pRight.Width / pRight.Height > 1.1);
                bool isSpread = (CheckSpread.IsChecked == true && !isAutoSingle);

                Mat? pLeft = (isSpread && index + 1 < _imageList.Count) ? LoadMat(_imageList[index + 1]) : null;
                _currentCombinedOriginal = CombineMats(pRight, pLeft);

                PageText.Text = isSpread ? $"P.{index + 2}-{index + 1} / {_imageList.Count}" : $"P.{index + 1} / {_imageList.Count}";

                if ((int)PageSlider.Value != index)
                {
                    _isUpdatingPageSliderInternal = true;
                    PageSlider.Value = index;
                    _isUpdatingPageSliderInternal = false;
                }

                UpdateImageDisplay();

                if (_onnxSession == null && !string.IsNullOrEmpty(_config.SelectedModel))
                {
                    WriteLog("Session missing on display. Attempting automatic reinit.");
                    InitializeAi(_config.SelectedModel);
                }

                if (_onnxSession != null && _inputName != null)
                {
                    StatusText.Text = $"Processing... ({_activeEngineMode})";
                    ShowAiProgressOsd();
                    var progress = new Progress<(int done, int total)>(p => UpdateAiProgressOsd(p.done, p.total));
                    var swTotal = System.Diagnostics.Stopwatch.StartNew();
                    try
                    {
                        var inferenceTask = Task.Run(() => PerformAiTiled(_currentCombinedOriginal, token, progress), token);
                        _currentInferenceTask = inferenceTask;

                        // 30秒ごとに「まだ動いているか」をログへ出し、真のハングと単なる低速処理を切り分けやすくする
                        while (true)
                        {
                            var completed = await Task.WhenAny(inferenceTask, Task.Delay(30000));
                            if (completed == inferenceTask || token.IsCancellationRequested) break;
                            WriteLog($"Still processing... elapsed {swTotal.Elapsed.TotalSeconds:F0}s (model: {_config.SelectedModel})");
                        }

                        _currentCombinedUpscaled = await inferenceTask;
                        if (!token.IsCancellationRequested)
                        {
                            WriteLog($"AI processing completed in {swTotal.Elapsed.TotalSeconds:F1}s.");
                            StatusText.Text = $"Mode: {_activeEngineMode}";
                            UpdateImageDisplay();
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        WriteLog($"AI processing cancelled after {swTotal.Elapsed.TotalSeconds:F1}s.");
                    }
                    catch (Exception ex)
                    {
                        WriteLog($"Runtime Kernel Error after {swTotal.Elapsed.TotalSeconds:F1}s (model: {_config.SelectedModel}): {ex}");
                        StatusText.Text = "Mode: Error (see session.log)";

                        string msg = ex.ToString();
                        if (msg.Contains("CUBLAS", StringComparison.OrdinalIgnoreCase) ||
                            msg.Contains("CUDA", StringComparison.OrdinalIgnoreCase) ||
                            msg.Contains("CUDNN", StringComparison.OrdinalIgnoreCase))
                        {
                            // GPUコンテキストが不安定化している可能性が高いため、
                            // 汚染されたセッションを使い回さず次回アクセス時に再構築させる
                            WriteLog("GPU runtime error detected. Disposing session to force clean reinit on next use.");
                            _onnxSession?.Dispose();
                            _onnxSession = null;
                            _inputName = null;
                        }
                    }
                    finally
                    {
                        _currentInferenceTask = null;
                        HideAiProgressOsd();
                    }
                }
            }
            catch (Exception ex) { WriteLog($"Display Abend: {ex.Message}"); }
        }

        private Mat LoadMat(string key)
        {
            if (Directory.Exists(_currentSourcePath)) return Cv2.ImRead(key);

            if (File.Exists(key) && IsImageFile(key))
            {
                return Cv2.ImRead(key);
            }

            string ext = Path.GetExtension(_currentSourcePath ?? "").ToLower(CultureInfo.InvariantCulture);
            if (ext == ".zip")
            {
                using (FileStream fs = new FileStream(_currentSourcePath!, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (ZipArchive archive = new ZipArchive(fs, ZipArchiveMode.Read))
                {
                    ZipArchiveEntry? entry = archive.GetEntry(key);
                    if (entry != null)
                    {
                        using (Stream entryStream = entry.Open())
                        using (MemoryStream ms = new MemoryStream())
                        {
                            entryStream.CopyTo(ms);
                            return Cv2.ImDecode(ms.ToArray(), ImreadModes.Color);
                        }
                    }
                }
            }
            else if (ext == ".rar" || ext == ".7z")
            {
                return LoadViaSystemCli(_currentSourcePath!, key);
            }

            return new Mat();
        }

        private List<string> GetArchiveFileListViaCli(string archivePath)
        {
            var files = new List<string>();
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "7z.exe",
                Arguments = $"l \"{archivePath}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                using (var proc = System.Diagnostics.Process.Start(startInfo))
                {
                    if (proc != null)
                    {
                        using (var reader = proc.StandardOutput)
                        {
                            string? line;
                            bool isDataSection = false;
                            while ((line = reader.ReadLine()) != null)
                            {
                                if (line.Contains("-------------------"))
                                {
                                    isDataSection = !isDataSection;
                                    continue;
                                }
                                if (isDataSection && line.Length > 53)
                                {
                                    string filename = line.Substring(53).Trim();
                                    if (IsImageFile(filename)) files.Add(filename);
                                }
                            }
                        }
                        proc.WaitForExit();
                    }
                }
            }
            catch (Exception ex) { WriteLog($"CLI List Error: {ex.Message}"); }
            return files.OrderBy(f => f).ToList();
        }

        private Mat LoadViaSystemCli(string archivePath, string entryKey)
        {
            if (!Directory.Exists(_tempExtractDir)) Directory.CreateDirectory(_tempExtractDir);

            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "7z.exe",
                Arguments = $"e \"{archivePath}\" -o\"{_tempExtractDir}\" \"{entryKey}\" -y",
                WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden,
                CreateNoWindow = true,
                UseShellExecute = false
            };

            try
            {
                using (var proc = System.Diagnostics.Process.Start(startInfo))
                {
                    proc?.WaitForExit();
                }

                string extractedPath = Path.Combine(_tempExtractDir, Path.GetFileName(entryKey));
                if (File.Exists(extractedPath))
                {
                    Mat mat = Cv2.ImRead(extractedPath);
                    try { File.Delete(extractedPath); } catch { }
                    return mat;
                }
            }
            catch (Exception ex) { WriteLog($"CLI Extracted Image Error: {ex.Message}"); }
            return new Mat();
        }

        private Mat CombineMats(Mat r, Mat? l)
        {
            if (l == null) return r.Clone();
            Mat res = new Mat(Math.Max(r.Height, l.Height), r.Width + l.Width, r.Type(), Scalar.Black);
            using (var roiL = new Mat(res, new OpenCvSharp.Rect(0, 0, l.Width, l.Height))) l.CopyTo(roiL);
            using (var roiR = new Mat(res, new OpenCvSharp.Rect(l.Width, 0, r.Width, r.Height))) r.CopyTo(roiR);
            r.Dispose(); l.Dispose();
            return res;
        }

        private class OsdPositionOption
        {
            public string Key = "";
            public string Label = "";
            public override string ToString() => Label;
        }

        private static readonly OsdPositionOption[] _osdPositions = new[]
        {
            new OsdPositionOption { Key = "TopCenter", Label = "上部中央" },
            new OsdPositionOption { Key = "TopLeft", Label = "左上" },
            new OsdPositionOption { Key = "TopRight", Label = "右上" },
            new OsdPositionOption { Key = "BottomCenter", Label = "下部中央（ボトムバー上）" },
        };

        private bool _isUpdatingOsdComboInternal = false;

        private void PopulateOsdPositionComboBox()
        {
            _isUpdatingOsdComboInternal = true;
            AiOsdPositionComboBox.ItemsSource = _osdPositions;
            AiOsdPositionComboBox.SelectedItem = _osdPositions.FirstOrDefault(o => o.Key == _config.AiOsdPosition) ?? _osdPositions[0];
            _isUpdatingOsdComboInternal = false;
        }

        private void AiOsdPositionComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (_isUpdatingOsdComboInternal) return;
            if (AiOsdPositionComboBox.SelectedItem is not OsdPositionOption option) return;

            _config.AiOsdPosition = option.Key;
            ApplyAiOsdPosition();
            SaveConfig();
        }

        private void ApplyAiOsdPosition()
        {
            switch (_config.AiOsdPosition)
            {
                case "TopLeft":
                    AiProgressOsd.HorizontalAlignment = HorizontalAlignment.Left;
                    AiProgressOsd.VerticalAlignment = VerticalAlignment.Top;
                    AiProgressOsd.Margin = new Thickness(20, 20, 0, 0);
                    break;
                case "TopRight":
                    AiProgressOsd.HorizontalAlignment = HorizontalAlignment.Right;
                    AiProgressOsd.VerticalAlignment = VerticalAlignment.Top;
                    AiProgressOsd.Margin = new Thickness(0, 20, 20, 0);
                    break;
                case "BottomCenter":
                    AiProgressOsd.HorizontalAlignment = HorizontalAlignment.Center;
                    AiProgressOsd.VerticalAlignment = VerticalAlignment.Bottom;
                    AiProgressOsd.Margin = new Thickness(0, 0, 0, 110);
                    break;
                case "TopCenter":
                default:
                    AiProgressOsd.HorizontalAlignment = HorizontalAlignment.Center;
                    AiProgressOsd.VerticalAlignment = VerticalAlignment.Top;
                    AiProgressOsd.Margin = new Thickness(0, 20, 0, 0);
                    break;
            }
        }

        private void ShowAiProgressOsd()
        {
            AiProgressBarFill.Width = 0;
            AiProgressText.Text = "AI変換中...";
            AiProgressOsd.Visibility = Visibility.Visible;
        }

        private void UpdateAiProgressOsd(int done, int total)
        {
            if (total <= 0) return;
            double ratio = Math.Clamp((double)done / total, 0, 1);
            AiProgressText.Text = $"AI変換中: {done} / {total} タイル ({(int)(ratio * 100)}%)";
            AiProgressBarFill.Width = AiProgressBarTrack.ActualWidth * ratio;
        }

        private void HideAiProgressOsd()
        {
            AiProgressOsd.Visibility = Visibility.Collapsed;
        }

        private void UpdateImageDisplay()
        {
            if (_currentCombinedOriginal == null) return;
            if (_currentCombinedUpscaled == null)
            {
                MainImage.Source = _currentCombinedOriginal.ToWriteableBitmap();
                return;
            }
            if (SplitSlider.Value >= 100)
            {
                MainImage.Source = _currentCombinedUpscaled.ToWriteableBitmap();
                return;
            }

            int w = _currentCombinedUpscaled.Width, h = _currentCombinedUpscaled.Height;
            int sx = (int)(w * (SplitSlider.Value / 100.0));
            Mat disp = new Mat(h, w, _currentCombinedUpscaled.Type());

            if (sx > 0)
            {
                using var roiAi = new Mat(disp, new OpenCvSharp.Rect(0, 0, sx, h));
                _currentCombinedUpscaled.SubMat(0, h, 0, sx).CopyTo(roiAi);
            }
            if (sx < w)
            {
                using var tmp = new Mat();
                Cv2.Resize(_currentCombinedOriginal, tmp, new OpenCvSharp.Size(w, h));
                using var roiRaw = new Mat(disp, new OpenCvSharp.Rect(sx, 0, w - sx, h));
                tmp.SubMat(0, h, sx, w).CopyTo(roiRaw);
            }
            if (sx > 25 && sx < w - 25)
            {
                Cv2.Rectangle(disp, new OpenCvSharp.Rect(sx - 25, 0, 50, h), new Scalar(0, 0, 255), -1);
            }

            MainImage.Source = disp.ToWriteableBitmap();
            disp.Dispose();
        }

        private void MovePage(int dir)
        {
            int step = (CheckSpread.IsChecked == true) ? 2 : 1;
            int next = (int)PageSlider.Value + (dir * step);
            if (next < 0) LoadNextArchive(-1);
            else if (next > PageSlider.Maximum) LoadNextArchive(1);
            else DisplayPage(next);
        }

        private void LoadNextArchive(int dir)
        {
            if (string.IsNullOrEmpty(_currentSourcePath)) return;
            var parent = Path.GetDirectoryName(_currentSourcePath);
            if (parent == null) return;
            var archives = Directory.GetFiles(parent, "*.*")
                .Where(f => f.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".rar", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".7z", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f).ToList();

            int idx = archives.IndexOf(_currentSourcePath) + dir;
            if (idx >= 0 && idx < archives.Count)
            {
                LoadSource(archives[idx]);
                if (dir < 0) DisplayPage((int)PageSlider.Maximum);
            }
        }

        private void ResetTransform()
        {
            ImgScale.ScaleX = 1.0;
            ImgScale.ScaleY = 1.0;
            ImgTranslate.X = 0;
            ImgTranslate.Y = 0;
        }
    }
}
