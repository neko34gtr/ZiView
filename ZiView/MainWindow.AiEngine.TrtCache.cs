using System;
using System.IO;
using System.Linq;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: TensorRTエンジンキャッシュの保存先解決、RAMDISK判定、
    /// および起動時/終了時のRAMDISK⇔SSD同期、全画面対応可否マーカーの読み書きを担当する。
    /// </summary>
    public partial class MainWindow
    {
        /// <summary>選択中モデル名に紐づく全画面対応可否マーカーのパス（trt_cacheディレクトリ内＝同期対象）。</summary>
        private string GetFullFrameMarkerPath()
        {
            string cacheDir = GetTensorRtCacheDirectory(_config.TensorRtCacheDirectory);
            string safeModelName = string.Join("_", _config.SelectedModel.Split(Path.GetInvalidFileNameChars()));
            return Path.Combine(cacheDir, $"{safeModelName}.fullframe_status");
        }

        private static string? ReadFullFrameMarker(string markerPath)
        {
            try
            {
                if (!File.Exists(markerPath)) return null;
                string content = File.ReadAllText(markerPath).Trim();
                return content == "supported" || content == "unsupported" ? content : null;
            }
            catch { return null; }
        }

        private static void WriteFullFrameMarker(string markerPath, string value)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(markerPath)!);
                File.WriteAllText(markerPath, value);
            }
            catch { /* 書き込めなくても次回また検証し直すだけなので致命的ではない */ }
        }

        /// <summary>
        /// TensorRTエンジンキャッシュの保存先。プログラムルート直下の固定フォルダとし、
        /// ModelFolderの変更に影響されないようにする。
        /// </summary>
        /// <summary>
        /// TensorRTエンジンキャッシュの出力先を決定する。
        /// customDirが指定されていればそれを最優先（書き込み確認込み）。
        /// 未指定時はX:\temp\ZView\trt_cache（RAMDISK運用想定）が使えればそちらへ、
        /// 使えなければプログラムルート直下\trt_cache へフォールバックする。
        /// </summary>
        internal static string GetTensorRtCacheDirectory(string? customDir)
        {
            if (!string.IsNullOrWhiteSpace(customDir))
            {
                try
                {
                    Directory.CreateDirectory(customDir);
                    return customDir;
                }
                catch
                {
                    // 指定フォルダに書き込めない場合は自動判定へフォールバックする
                }
            }

            try
            {
                if (Directory.Exists(@"X:\"))
                {
                    const string ramdiskCacheDir = @"X:\temp\ZView\trt_cache";
                    Directory.CreateDirectory(ramdiskCacheDir);
                    return ramdiskCacheDir;
                }
            }
            catch { }

            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "trt_cache");
        }

        // SSD側の退避先（プログラムルート直下・固定）。RAMDISK運用時のみ実際に使われる。
        private static string GetTrtCacheBackupDirectory()
            => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "trt_cache_backup");

        private const long CacheSyncMinFreeBytes = 512L * 1024 * 1024; // 512MB

        /// <summary>
        /// パスが指すドライブがRAMDISK（DriveType.Ram）かどうかを自動判定する。
        /// ドライブレターだけでは判別できないため、DriveInfoのDriveTypeで判定する。
        /// 一部のRAMDISKドライバはFixed扱いで報告してくることがあり、その場合は誤ってfalseになる
        /// （＝同期をスキップするだけで安全側に倒れる。手動フラグでの上書きが必要ならSettings側の対応が別途必要）。
        /// </summary>
        private static bool IsRamDisk(string path)
        {
            try
            {
                string? root = Path.GetPathRoot(Path.GetFullPath(path));
                if (string.IsNullOrEmpty(root)) return false;
                var drive = new DriveInfo(root);
                return drive.IsReady && drive.DriveType == DriveType.Ram;
            }
            catch { return false; }
        }

        /// <summary>
        /// 設定（TrtCacheRamDiskOverride）を優先してRAMDISK扱いかどうかを判定する。
        /// "Auto"の場合のみDriveTypeによる自動判定（IsRamDisk）にフォールバックする。
        /// 一部のRAMDISKドライバはFixed扱いで報告され自動判定が外れることがあるため、
        /// SettingsWindowから強制指定できるようにしてある。
        /// </summary>
        private bool IsRamDiskForCache(string path)
        {
            return _config.TrtCacheRamDiskOverride switch
            {
                "ForceOn" => true,
                "ForceOff" => false,
                _ => IsRamDisk(path),
            };
        }

        private static long GetFreeBytesForPath(string path)
        {
            try
            {
                string? root = Path.GetPathRoot(Path.GetFullPath(path));
                if (string.IsNullOrEmpty(root)) return -1;
                return new DriveInfo(root).AvailableFreeSpace;
            }
            catch { return -1; }
        }

        /// <summary>
        /// srcDir配下の全ファイルをdstDirへ相対パスを保ったままファイル単位で上書きコピーする。
        /// ディレクトリ丸ごと削除・差し替えは行わない（誤動作時に意図しないフォルダを消す事故を避けるため）。
        /// </summary>
        private static int CopyDirectoryFiles(string srcDir, string dstDir)
        {
            int count = 0;
            foreach (var srcFile in Directory.EnumerateFiles(srcDir, "*", SearchOption.AllDirectories))
            {
                string rel = Path.GetRelativePath(srcDir, srcFile);
                string dstFile = Path.Combine(dstDir, rel);
                string? dstFileDir = Path.GetDirectoryName(dstFile);
                if (!string.IsNullOrEmpty(dstFileDir)) Directory.CreateDirectory(dstFileDir);
                File.Copy(srcFile, dstFile, overwrite: true);
                count++;
            }
            return count;
        }

        /// <summary>
        /// 起動時（InitializeAiの前）に呼ぶ。TensorRTキャッシュ先がRAMDISKで、かつ中身が空
        /// （PC再起動でRAMDISKが消えた状態）の場合のみ、SSD側の退避フォルダから復元する。
        /// RAMDISK側の空き容量が「復元に必要なサイズ+512MB」未満なら安全側でスキップする。
        /// </summary>
        private void RestoreTrtCacheFromBackupIfNeeded()
        {
            try
            {
                string liveCacheDir = GetTensorRtCacheDirectory(_config.TensorRtCacheDirectory);
                if (!IsRamDiskForCache(liveCacheDir))
                {
                    WriteLog("[Cache] TRT cache is not on a RAM disk; skip restore/backup sync.");
                    return;
                }

                bool liveHasFiles = Directory.Exists(liveCacheDir)
                    && Directory.EnumerateFiles(liveCacheDir, "*", SearchOption.AllDirectories).Any();
                if (liveHasFiles)
                {
                    WriteLog("[Cache] RAM disk TRT cache already present; no restore needed.");
                    return;
                }

                string backupDir = GetTrtCacheBackupDirectory();
                if (!Directory.Exists(backupDir)
                    || !Directory.EnumerateFiles(backupDir, "*", SearchOption.AllDirectories).Any())
                {
                    WriteLog("[Cache] No SSD backup TRT cache found; will build fresh.");
                    return;
                }

                long backupSize = new DirectoryInfo(backupDir)
                    .EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
                long freeOnRamDisk = GetFreeBytesForPath(liveCacheDir);
                string resolvedRoot = Path.GetPathRoot(Path.GetFullPath(liveCacheDir)) ?? "?";
                if (freeOnRamDisk >= 0 && freeOnRamDisk < backupSize + CacheSyncMinFreeBytes)
                {
                    WriteLog($"[Cache] RAM disk free space insufficient for restore " +
                             $"({freeOnRamDisk / 1024 / 1024}MB free on {resolvedRoot}, need {(backupSize + CacheSyncMinFreeBytes) / 1024 / 1024}MB, path={liveCacheDir}). Skipping.");
                    return;
                }

                var sw = System.Diagnostics.Stopwatch.StartNew();
                Directory.CreateDirectory(liveCacheDir);
                int copied = CopyDirectoryFiles(backupDir, liveCacheDir);
                WriteLog($"[Cache] Restored {copied} TRT cache file(s) from SSD backup to RAM disk in {sw.Elapsed.TotalMilliseconds:F0}ms.");
            }
            catch (Exception ex)
            {
                WriteLog($"[Cache] TRT cache restore failed (continuing without it): {ex.Message}");
            }
        }

        /// <summary>
        /// 終了時（Window_Closing）に呼ぶ。TensorRTキャッシュ先がRAMDISKの場合のみ、
        /// SSD側の退避フォルダへファイル単位で上書き同期する。SSD側の空き容量が
        /// 「退避に必要なサイズ+512MB」未満なら安全側でスキップする。
        /// </summary>
        private void BackupTrtCacheToSsd()
        {
            try
            {
                string liveCacheDir = GetTensorRtCacheDirectory(_config.TensorRtCacheDirectory);
                if (!IsRamDiskForCache(liveCacheDir)) return; // RAMDISK運用時のみ退避が必要

                if (!Directory.Exists(liveCacheDir)) return;
                var liveFiles = Directory.EnumerateFiles(liveCacheDir, "*", SearchOption.AllDirectories).ToList();
                if (liveFiles.Count == 0) return;

                string backupDir = GetTrtCacheBackupDirectory();
                long liveSize = liveFiles.Sum(f => new FileInfo(f).Length);
                long freeOnSsd = GetFreeBytesForPath(backupDir);
                string resolvedRoot = Path.GetPathRoot(Path.GetFullPath(backupDir)) ?? "?";
                if (freeOnSsd >= 0 && freeOnSsd < liveSize + CacheSyncMinFreeBytes)
                {
                    WriteLog($"[Cache] SSD free space insufficient for backup " +
                             $"({freeOnSsd / 1024 / 1024}MB free on {resolvedRoot}, need {(liveSize + CacheSyncMinFreeBytes) / 1024 / 1024}MB, path={backupDir}). Skipping.");
                    return;
                }

                Directory.CreateDirectory(backupDir);
                int copied = CopyDirectoryFiles(liveCacheDir, backupDir);
                WriteLog($"[Cache] Backed up {copied} TRT cache file(s) from RAM disk to SSD.");
            }
            catch (Exception ex)
            {
                WriteLog($"[Cache] TRT cache backup failed: {ex.Message}");
            }
        }
    }
}
