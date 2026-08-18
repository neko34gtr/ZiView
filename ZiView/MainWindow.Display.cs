using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using ImageMagick;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial（コア）: 画像/アーカイブソースの読み込み、ページ送り、
    /// 見開き合成、AI比較表示（分割スライダー）、AI進捗OSDを担当する。
    /// トーンカーブLUTの適用（ApplyToneCurveForDisplay）はMainWindow.ToneCurve.cs、
    /// 見開き境界の追跡（_currentSpreadLeftWidth等）と片側ページ位置微調整は
    /// MainWindow.PageOffset.cs、回転状態のリセットはMainWindow.Rotation.csのフィールドを参照。
    /// </summary>
    public partial class MainWindow
    {
        private string? _currentSourcePath;
        private List<string> _imageList = new List<string>();
        private Mat? _currentCombinedOriginal;
        private Mat? _currentCombinedUpscaled;
        private readonly string _tempExtractDir = Path.Combine(Path.GetTempPath(), "ZiView_Temp");

        // AI先読み（プリフェッチ）用: 事前デコード・事前AI推論済みのページを一時保持するキャッシュ。
        // キー=ページインデックス。現在ページの表示完了直後に「次ページ」のみを1件先読みする軽量実装。
        private readonly Dictionary<int, (Mat Original, Mat? Upscaled, bool IsSpread, int LeftWidth)> _pageCache = new();
        private CancellationTokenSource? _prefetchCts;
        private Task? _prefetchTask;

        // 無限ループイベントを抑止するためのフラグ
        private bool _isUpdatingPageSliderInternal = false;

        /// <summary>
        /// 先読みキャッシュを全て破棄する。ソース切替・モデル切替・見開き設定変更など、
        /// キャッシュ内容の前提が崩れるタイミングで必ず呼び出すこと。
        /// </summary>
        private void ClearPageCache()
        {
            _prefetchCts?.Cancel();
            foreach (var entry in _pageCache.Values)
            {
                entry.Original.Dispose();
                entry.Upscaled?.Dispose();
            }
            _pageCache.Clear();
        }

        // 現在のソースがzipの場合、ページ送りのたびに開き直さずこのハンドルを使い回す
        private ZipArchive? _openZipArchive;
        private readonly object _zipReadLock = new object();
        private FileStream? _openZipStream;

        // AVIFデコード失敗時にImageMagickがネイティブ側へ残す一時ファイル([TEMP]\magick-*)を
        // 自前管理のフォルダに閉じ込め、起動時に前回セッションの残骸をまとめて掃除できるようにする。
        private static readonly string _magickTempDir = Path.Combine(Path.GetTempPath(), "ZiView", "magick-cache");

        /// <summary>
        /// アプリ起動時に一度だけ呼ぶ。ImageMagickの一時ディレクトリをOS既定から
        /// _magickTempDirへ切り替え、前回セッションが残した孤児ファイルを削除する。
        /// AVIFデコード（MagickImageのコンストラクタ）が例外を投げた場合、usingでの
        /// Disposeが効かず一時ファイルが残り続けることがあるための対策。
        /// </summary>
        internal static void InitializeMagickTempDirectory()
        {
            try
            {
                Directory.CreateDirectory(_magickTempDir);
                MagickNET.SetTempDirectory(_magickTempDir);
            }
            catch
            {
                // 失敗してもOS既定の一時フォルダが使われるだけなので致命的ではない
            }

            try
            {
                foreach (var f in Directory.EnumerateFiles(_magickTempDir))
                {
                    try { File.Delete(f); }
                    catch { /* 前回セッションがまだファイルをロックしている等は次回に回す */ }
                }
            }
            catch { }
        }

        /// <summary>
        /// 開いたままのzipハンドルを解放する。ソース切替時・終了時に必ず呼び出すこと。
        /// </summary>
        private void CloseOpenZip()
        {
            _openZipArchive?.Dispose();
            _openZipArchive = null;
            _openZipStream?.Dispose();
            _openZipStream = null;
        }


        public async Task LoadSource(string path)
        {
            WriteLog($"Loading source tree: {path}");

            _cts?.Cancel();

            _prefetchCts?.Cancel(); // ClearPageCache内でも呼ばれるが、下のawaitより前に確実に要求しておく

            //（旧ソースの_openZipArchiveをDisposeする前に、それを参照している
            // かもしれないバックグラウンドタスク（推論・先読み）の完了を待つ。DisplayPageと同じ理由）
            var priorTask = _currentInferenceTask;
            if (priorTask != null)
            {
                try { await priorTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }
            var priorPrefetch = _prefetchTask;
            if (priorPrefetch != null)
            {
                try { await priorPrefetch; } catch { /* キャンセル/実行時例外は無視 */ }
            }

            ClearPageCache();
            CloseOpenZip();

            _currentSourcePath = path;
            _imageList.Clear();
            int initialIndex = 0;
            try
            {
                if (Directory.Exists(path))
                {
                    var files = Directory.GetFiles(path)
                        .Where(f => !string.IsNullOrEmpty(f) && IsImageFile(f))
                        .OrderBy(f => f, NaturalStringComparer.Instance);
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
                                .OrderBy(f => f, NaturalStringComparer.Instance);
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
                            // ページ送りのたびにFileStream/ZipArchiveを開き直していたのをやめ、
                            // このソースを閉じるまで開いたまま保持して使い回す（I/O削減）
                            _openZipStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                            _openZipArchive = new ZipArchive(_openZipStream, ZipArchiveMode.Read);

                            var keys = _openZipArchive.Entries
                                .Where(e => !string.IsNullOrEmpty(e.FullName) && IsImageFile(e.FullName))
                                .Select(e => e.FullName)
                                .OrderBy(k => k, NaturalStringComparer.Instance);
                            _imageList.AddRange(keys);
                        }
                        else if (ext == ".rar" || ext == ".7z")
                        {
                            // ページ毎にCLIへ都度アクセスする方式（文字コード変換を伴い文字化けの原因になりやすい）
                            // をやめ、起動時に実体としてディスクへ全展開する。以降はNTFS上の実ファイルパスを
                            // そのまま使うため、エントリ名の文字化けが原理的に発生しない。
                            var files = await ExtractArchiveAsync(path);
                            _imageList.AddRange(files);
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
            _ = LoadSource(path);
        }

        private bool IsImageFile(string? f) =>
            !string.IsNullOrEmpty(f) && (
            f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".webp", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".avif", StringComparison.OrdinalIgnoreCase));

        private void RefreshDisplay() => DisplayPage((int)PageSlider.Value);

        private async void DisplayPage(int index)
        {
            _cts?.Cancel();
            _prefetchCts?.Cancel();

            var priorTask = _currentInferenceTask;
            if (priorTask != null)
            {
                // 前ページの推論が_currentCombinedOriginal等を参照中の可能性があるため、
                // 完全に終わってから破棄する（ObjectDisposedException対策）
                try { await priorTask; } catch { /* キャンセル/実行時例外は無視 */ }
            }
            var priorPrefetch = _prefetchTask;
            if (priorPrefetch != null)
            {
                // 先読みタスクが_pageCacheへ書き込み中の可能性があるため、完了を待ってから続行する
                try { await priorPrefetch; } catch { /* キャンセル/実行時例外は無視 */ }
            }

            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            if (index < 0 || index >= _imageList.Count) return;

            try
            {
                bool isSpread;

                // 下部ページスライダーの操作中（ホバー/ドラッグ中）は、すっ飛ばし移動が前提のため
                // AI推論そのものを起こさない。無駄な推論の連鎖キャンセル・キュー暴走を防ぐのが目的で、
                // 操作をやめた瞬間（ホバー解除かつドラッグ解放）に改めてAI推論付きで再表示する
                bool sliderBusy = PageSlider.IsMouseOver || PageSlider.IsMouseCaptureWithin;

                if (_pageCache.TryGetValue(index, out var cached))
                {
                    // 先読み済みのページはデコード・AI推論を一切行わず即座に反映する
                    _pageCache.Remove(index);
                    _currentCombinedOriginal?.Dispose();
                    _currentCombinedUpscaled?.Dispose();
                    _currentCombinedOriginal = cached.Original;
                    _currentCombinedUpscaled = cached.Upscaled;
                    isSpread = cached.IsSpread;

                    bool hasNextCached = index + 1 < _imageList.Count;
                    bool isLeftOpenCached = _readingMode == PageOpenMode.LeftOpen;
                    _currentSpreadLeftWidth = (isSpread && hasNextCached) ? cached.LeftWidth : 0;
                    _currentSpreadLeftKey = (isSpread && hasNextCached) ? (isLeftOpenCached ? _imageList[index] : _imageList[index + 1]) : null;
                    _currentSpreadRightKey = (isSpread && hasNextCached) ? (isLeftOpenCached ? _imageList[index + 1] : _imageList[index]) : _imageList[index];

                    int leftPageNumCached = isLeftOpenCached ? index + 1 : index + 2;
                    int rightPageNumCached = isLeftOpenCached ? index + 2 : index + 1;
                    PageText.Text = isSpread ? $"P.{leftPageNumCached}-{rightPageNumCached} / {_imageList.Count}" : $"P.{index + 1} / {_imageList.Count}";
                    if ((int)PageSlider.Value != index)
                    {
                        _isUpdatingPageSliderInternal = true;
                        PageSlider.Value = index;
                        _isUpdatingPageSliderInternal = false;
                    }

                    UpdateImageDisplay();
                    StatusText.Text = _currentCombinedUpscaled != null ? $"Mode: {_activeEngineMode} (先読み済み)" : "Mode: AI Disabled";
                }
                else
                {
                    _currentCombinedOriginal?.Dispose();
                    _currentCombinedUpscaled?.Dispose();
                    _currentCombinedUpscaled = null;

                    bool autoDetectEnabled = CheckAutoDetect.IsChecked == true;

                    // 画像デコード（ファイルI/O・展開）はUIスレッドをブロックしないようバックグラウンドで実行する
                    var (combined, spread, leftWidth) = await Task.Run(() => DecodeCombinedPage(index, _readingMode, autoDetectEnabled), token);
                    if (token.IsCancellationRequested) { combined.Dispose(); return; }

                    _currentCombinedOriginal = combined;
                    isSpread = spread;

                    bool hasNextFresh = index + 1 < _imageList.Count;
                    bool isLeftOpenFresh = _readingMode == PageOpenMode.LeftOpen;
                    _currentSpreadLeftWidth = (isSpread && hasNextFresh) ? leftWidth : 0;
                    _currentSpreadLeftKey = (isSpread && hasNextFresh) ? (isLeftOpenFresh ? _imageList[index] : _imageList[index + 1]) : null;
                    _currentSpreadRightKey = (isSpread && hasNextFresh) ? (isLeftOpenFresh ? _imageList[index + 1] : _imageList[index]) : _imageList[index];

                    int leftPageNumFresh = isLeftOpenFresh ? index + 1 : index + 2;
                    int rightPageNumFresh = isLeftOpenFresh ? index + 2 : index + 1;
                    PageText.Text = isSpread ? $"P.{leftPageNumFresh}-{rightPageNumFresh} / {_imageList.Count}" : $"P.{index + 1} / {_imageList.Count}";

                    if ((int)PageSlider.Value != index)
                    {
                        _isUpdatingPageSliderInternal = true;
                        PageSlider.Value = index;
                        _isUpdatingPageSliderInternal = false;
                    }

                    UpdateImageDisplay();

                    if (sliderBusy)
                    {
                        StatusText.Text = "Mode: Skipped (ページ送り中)";
                    }
                    else if (!_config.EnableAiInference)
                    {
                        StatusText.Text = "Mode: AI Disabled";
                    }
                    else
                    {
                        if (_onnxSession == null && !string.IsNullOrEmpty(_config.SelectedModel))
                        {
                            WriteLog("Session missing on display. Attempting automatic reinit.");
                            // 同期呼び出しだとUIスレッドをTensorRT構築完了までブロックしてしまう上、
                            // 起動直後の初期化とほぼ同時に走ると二重にVRAMを取り合いOOMを誘発したことがあるため、
                            // オーバーレイ付き非同期版（内部で多重実行防止ガード済み）を使う。
                            await InitializeAiWithOverlayAsync(_config.SelectedModel);
                        }

                        // AI推論実行部を、縮小処理を通してから渡す
                        if (_onnxSession != null && _inputName != null)
                        {
                            StatusText.Text = $"Processing... ({_activeEngineMode})";
                            ShowAiProgressOsd();
                            var progress = new Progress<(int done, int total)>(p => UpdateAiProgressOsd(p.done, p.total));
                            var swTotal = System.Diagnostics.Stopwatch.StartNew();
                            try
                            {
                                var inferenceTask = Task.Run(() =>
                                {
                                    // 巨大画像によるVRAMパンクを防ぐため、長辺3840px上限で推論用Matを作成
                                    using var inputForAi = PrepareMatForInference(_currentCombinedOriginal, 3840);
                                    return PerformAiTiled(inputForAi, token, progress);
                                }, token);

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
                }

                // 現在ページの表示が完了した直後、AI先読みが有効な場合のみ次ページ以降を裏で準備しておく
                // （スライダー操作中はどうせ通り過ぎるだけなので先読み自体も無駄になるため起こさない）
                if (!sliderBusy && CheckPrefetch.IsChecked == true && !token.IsCancellationRequested)
                {
                    int step = (_readingMode != PageOpenMode.Single) ? 2 : 1;
                    int count = Math.Clamp(_config.PrefetchPageCount, 1, 5);
                    _prefetchCts = new CancellationTokenSource();
                    _prefetchTask = PrefetchAheadAsync(index + step, step, count, _prefetchCts.Token);
                }
            }
            catch (OperationCanceledException)
            {
                // 新しいページ遷移によるキャンセルは正常系のため、エラーとしては扱わない
            }
            catch (Exception ex) { WriteLog($"Display Abend: {ex.Message}"); }
        }

        /// <summary>
        /// 指定ページの画像をデコードし（必要なら見開き合成した上で）返す。
        /// ファイルI/Oやアーカイブ展開を伴うため、呼び出し側はTask.Run等でUIスレッド外から呼ぶこと。
        /// mode/autoDetectEnabledはUI要素へアクセスせずに済むよう、呼び出し元でUIスレッド上から読み取って渡す。
        /// modeがRightOpen（右開き＝従来のマンガ方式）なら現在ページ(index)を右、次ページ(index+1)を左に、
        /// LeftOpen（左開き＝洋書方式）なら現在ページを左、次ページを右に配置する。
        /// LeftWidthは見開き合成時の左ページの幅（＝合成画像内での左右の境界X座標）。非見開き時は0。
        /// 見開き時の片側ページ位置微調整機能（MainWindow.PageOffset.cs）が、合成後のビットマップ上で
        /// どこからどこまでが左ページ/右ページかを判定するために使う。
        /// </summary>
        private (Mat Combined, bool IsSpread, int LeftWidth) DecodeCombinedPage(int index, PageOpenMode mode, bool autoDetectEnabled)
        {
            Mat pCurrent = LoadMat(_imageList[index]);
            bool isAutoSingle = (autoDetectEnabled && pCurrent.Height > 0 && (double)pCurrent.Width / pCurrent.Height > 1.1);
            bool isSpread = (mode != PageOpenMode.Single && !isAutoSingle);

            Mat? pOther = (isSpread && index + 1 < _imageList.Count) ? LoadMat(_imageList[index + 1]) : null;

            if (pOther == null)
            {
                // ペア相手がいない（設定上は見開きでもアーカイブ最終ページ等）場合は単ページのまま返す
                return (pCurrent, isSpread, 0);
            }

            Mat combined;
            int leftWidth;

            // 重要: 幅は必ずCombineMats呼び出しより前に取得すること。CombineMatsは内部でr/l両方を
            // Disposeするため、呼び出し後にpCurrent/pOther.Widthへアクセスするとuse-after-disposeとなり、
            // OpenCvSharpのMat.Widthはネイティブポインタを直接参照するため、.NET例外にならず
            // アプリごと無言でクラッシュする（実際に発生していた不具合）。
            if (mode == PageOpenMode.LeftOpen)
            {
                leftWidth = pCurrent.Width;
                combined = CombineMats(pOther, pCurrent); // CombineMats(r, l): rを右、lを左に配置
            }
            else
            {
                leftWidth = pOther.Width;
                combined = CombineMats(pCurrent, pOther);
            }

            return (combined, isSpread, leftWidth);
        }

        /// <summary>
        /// 先読み設定件数(count)ぶん、startIndexからstepおきに順番に先読みしていく。
        /// GPU/推論セッションへの負荷を予測しやすくするため、あえて並列化せず1ページずつ逐次実行する。
        /// 既にキャッシュ済み・範囲外のページはスキップする。
        /// </summary>
        private async Task PrefetchAheadAsync(int startIndex, int step, int count, CancellationToken token)
        {
            int idx = startIndex;
            for (int i = 0; i < count; i++)
            {
                if (token.IsCancellationRequested) return;
                if (idx < 0 || idx >= _imageList.Count) break;
                await PrefetchOnePageAsync(idx, token);
                idx += step;
            }
        }

        /// <summary>
        /// 「AI先読み」有効時に、1ページ分だけバックグラウンドで事前デコード・事前AI推論しておく。
        /// 結果は_pageCacheへ格納し、実際にそのページへ遷移した際はデコード・推論を丸ごとスキップして
        /// 即座に表示できるようにする。
        /// </summary>
        private async Task PrefetchOnePageAsync(int index, CancellationToken token)
        {
            if (index < 0 || index >= _imageList.Count) return;
            if (_pageCache.ContainsKey(index)) return;

            // UIスレッド上にいる間（最初のawait前）に現在の開き方向・自動判定設定を読み取っておく
            PageOpenMode mode = _readingMode;
            bool autoDetectEnabled = CheckAutoDetect.IsChecked == true;
            bool aiEnabled = _config.EnableAiInference && _onnxSession != null && _inputName != null;

            Mat? combined = null;
            Mat? upscaled = null;
            try
            {
                var (decoded, isSpread, leftWidth) = await Task.Run(() => DecodeCombinedPage(index, mode, autoDetectEnabled), token);
                combined = decoded;
                token.ThrowIfCancellationRequested();

                if (aiEnabled)
                {
                    upscaled = await Task.Run(() =>
                    {
                        // 巨大画像によるVRAMパンクを防ぐため、長辺3840px上限で推論用Matを作成
                        using var inputForAi = PrepareMatForInference(combined, 3840);
                        return PerformAiTiled(inputForAi, token);
                    }, token);
                    token.ThrowIfCancellationRequested();
                }

                if (_pageCache.ContainsKey(index))
                {
                    // 稀に競合した場合は先勝ちを残し、こちらは破棄する
                    combined.Dispose();
                    upscaled?.Dispose();
                    return;
                }

                _pageCache[index] = (combined, upscaled, isSpread, leftWidth);
                WriteLog($"[Prefetch] Page {index + 1} ready in background.");
            }
            catch (OperationCanceledException)
            {
                // ページ送り・設定変更等によるキャンセルは正常系
                combined?.Dispose();
                upscaled?.Dispose();
            }
            catch (Exception ex)
            {
                WriteLog($"Prefetch Error: {ex.Message}");
                combined?.Dispose();
                upscaled?.Dispose();
            }
        }

        private Mat LoadMat(string key)
        {
            if (Directory.Exists(_currentSourcePath)) return LoadMatFromPath(key);

            // rar/7zは起動時に実ファイルとして全展開済みのため、keyは既にディスク上の実パスになっている
            if (File.Exists(key) && IsImageFile(key))
            {
                return LoadMatFromPath(key);
            }

            string ext = Path.GetExtension(_currentSourcePath ?? "").ToLower(CultureInfo.InvariantCulture);
            if (ext == ".zip" && _openZipArchive != null)
            {
                // ZipArchiveは複数エントリの同時読み取りに非対応のため、
                // 先読みスレッドと表示スレッドが同時にここへ来ても直列化する）
                lock (_zipReadLock)
                {
                    ZipArchiveEntry? entry = _openZipArchive.GetEntry(key);
                    if (entry != null)
                    {
                        using (Stream entryStream = entry.Open())
                        using (MemoryStream ms = new MemoryStream())
                        {
                            entryStream.CopyTo(ms);
                            return DecodeImageBytes(ms.ToArray(), key);
                        }
                    }
                }
            }

            return new Mat();
        }

        /// <summary>
        /// ディスク上の実ファイルパスからMatを読み込む。AVIFのみOpenCVがデコード非対応のため
        /// Magick.NET経由でバイト列化してからCv2.ImDecodeへ渡す。それ以外は従来通りCv2.ImReadを使う。
        /// </summary>
        private Mat LoadMatFromPath(string path)
        {
            string ext = Path.GetExtension(path).ToLower(CultureInfo.InvariantCulture);
            if (ext == ".avif")
            {
                return DecodeImageBytes(File.ReadAllBytes(path), path);
            }
            return Cv2.ImRead(path);
        }

        /// <summary>
        /// AVIFはOpenCVが未対応のためMagick.NETでデコードしBMPバイト列へ変換してからCv2.ImDecodeへ渡す。
        /// ImreadModes.Colorで強制的に3ch BGRへ揃えるため、アルファの有無に関わらず他形式と同じ扱いになる。
        /// それ以外の拡張子は従来通りCv2.ImDecodeへそのまま渡す。
        /// </summary>
        private Mat DecodeImageBytes(byte[] bytes, string keyOrPath)
        {
            string ext = Path.GetExtension(keyOrPath).ToLower(CultureInfo.InvariantCulture);
            if (ext == ".avif")
            {
                using var magick = new MagickImage(bytes);
                byte[] bmpBytes = magick.ToByteArray(MagickFormat.Bmp);
                return Cv2.ImDecode(bmpBytes, ImreadModes.Color);
            }
            return Cv2.ImDecode(bytes, ImreadModes.Color);
        }

        /// <summary>
        /// rar/7zアーカイブを一時フォルダへ丸ごと展開し、その中の画像ファイルパス一覧を返す。
        /// 従来はページ送りのたびに"7z l"のテキスト出力をカラム位置で解析しており、
        /// コンソールの文字コード変換を経由するため日本語ファイル名が文字化けしやすかった。
        /// 本方式ではファイル名の解釈を一切行わず、7z.exeが直接NTFS上へ書き出した実ファイル名を
        /// そのまま使うため、エントリ名の文字化けが原理的に発生しない。副次的に、ページ送り毎の
        /// プロセス起動コストも初回展開の1回分に集約され、体感速度が大きく改善する。
        /// </summary>
        private async Task<List<string>> ExtractArchiveAsync(string archivePath)
        {
            try
            {
                if (Directory.Exists(_tempExtractDir)) Directory.Delete(_tempExtractDir, true);
            }
            catch (Exception ex) { WriteLog($"Temp Extract Cleanup Error: {ex.Message}"); }

            try
            {
                Directory.CreateDirectory(_tempExtractDir);
            }
            catch (Exception ex)
            {
                WriteLog($"Temp Extract Dir Create Error: {ex.Message}");
                return new List<string>();
            }

            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "7z.exe",
                // -y: 上書き確認を省略。ディレクトリ構造を保持したまま丸ごと展開する
                Arguments = $"x \"{archivePath}\" -o\"{_tempExtractDir}\" -y",
                WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden,
                CreateNoWindow = true,
                UseShellExecute = false
            };

            try
            {
                using var proc = System.Diagnostics.Process.Start(startInfo);
                if (proc != null) await proc.WaitForExitAsync();
            }
            catch (Exception ex)
            {
                WriteLog($"Archive Extract Error: {ex.Message}");
                return new List<string>();
            }

            try
            {
                return Directory.GetFiles(_tempExtractDir, "*", SearchOption.AllDirectories)
                    .Where(f => IsImageFile(f))
                    .OrderBy(f => f, NaturalStringComparer.Instance)
                    .ToList();
            }
            catch (Exception ex)
            {
                WriteLog($"Extracted File Enumeration Error: {ex.Message}");
                return new List<string>();
            }
        }

        private Mat CombineMats(Mat r, Mat? l)
        {
            // 単ページ表示時はrの所有権をそのまま結果へ引き継ぐ（不要なCloneと、それに伴うrの破棄漏れを解消）
            if (l == null) return r;

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

        /// <summary>
        /// トーンカーブ(Tキー)のLUT適用（ApplyToneCurveForDisplay）はMainWindow.ToneCurve.cs側で定義。
        /// </summary>
        private void UpdateImageDisplay()
        {
            if (_currentCombinedOriginal == null) return;
            if (_currentCombinedUpscaled == null)
            {
                var toned = ApplyToneCurveForDisplay(_currentCombinedOriginal);
                MainImage.Source = toned.ToWriteableBitmap();
                if (toned != _currentCombinedOriginal) toned.Dispose();
                RefreshPageOffsetOverlays();
                return;
            }
            if (SplitSlider.Value >= 100)
            {
                var toned = ApplyToneCurveForDisplay(_currentCombinedUpscaled);
                MainImage.Source = toned.ToWriteableBitmap();
                if (toned != _currentCombinedUpscaled) toned.Dispose();
                RefreshPageOffsetOverlays();
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

            var finalDisp = ApplyToneCurveForDisplay(disp);
            MainImage.Source = finalDisp.ToWriteableBitmap();
            if (finalDisp != disp) finalDisp.Dispose();
            disp.Dispose();
            RefreshPageOffsetOverlays();
        }

        private void MovePage(int dir)
        {
            int step = (_readingMode != PageOpenMode.Single) ? 2 : 1;
            int next = (int)PageSlider.Value + (dir * step);
            if (next < 0) LoadNextArchive(-1);
            else if (next > PageSlider.Maximum) LoadNextArchive(1);
            else DisplayPage(next);
        }

        private async void LoadNextArchive(int dir)
        {
            if (string.IsNullOrEmpty(_currentSourcePath)) return;
            var parent = Path.GetDirectoryName(_currentSourcePath);
            if (parent == null) return;
            var archives = Directory.GetFiles(parent, "*.*")
                .Where(f => f.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".rar", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".7z", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f, NaturalStringComparer.Instance).ToList();

            int idx = archives.IndexOf(_currentSourcePath) + dir;
            if (idx >= 0 && idx < archives.Count)
            {
                // rar/7zは全展開を伴い時間がかかるため、完了を待ってからページ位置を決定する
                // （待たずに進むとPageSlider.Maximumが旧アーカイブの値のままになり遷移がずれる）
                await LoadSource(archives[idx]);
                if (dir < 0) DisplayPage((int)PageSlider.Maximum);
            }
        }

        private void ResetTransform()
        {
            ImgScale.ScaleX = 1.0;
            ImgScale.ScaleY = 1.0;
            ImgTranslate.X = 0;
            ImgTranslate.Y = 0;

            // 表示回転（Rキー）は別ソース読込時にリセットする（設定として永続化はしていない）
            _rotationAngle = 0;
            if (_imgRotate != null) _imgRotate.Angle = 0;
        }
    }
}
