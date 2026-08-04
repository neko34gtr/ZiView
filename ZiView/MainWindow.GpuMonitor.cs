using System;
using System.Runtime.InteropServices;

namespace ZiView
{
    /// <summary>
    /// MainWindow partial: NVIDIA GPU使用率・VRAM表示（NVMLのP/Invoke）とRAM使用量表示を担当する。
    /// </summary>
    public partial class MainWindow
    {
        // --- NVIDIA NVML P/Invoke Definitions ---
        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlInit();

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlShutdown();

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlDeviceGetHandleByIndex(uint index, out IntPtr device);

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlDeviceGetMemoryInfo(IntPtr device, out NVML_MEMORY memory);

        [DllImport("nvml.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        private static extern int nvmlDeviceGetUtilizationRates(IntPtr device, out NVML_UTILIZATION utilization);

        [StructLayout(LayoutKind.Sequential)]
        private struct NVML_MEMORY
        {
            public ulong total;
            public ulong free;
            public ulong used;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct NVML_UTILIZATION
        {
            public uint gpu;
            public uint memory;
        }

        private bool _isNvidiaGpu = false;
        private bool _nvmlInitialized = false;
        private bool _isGpuCheckAttempted = false;
        private IntPtr _nvmlDevice = IntPtr.Zero;

        private void InitGpuMonitor()
        {
            if (_isGpuCheckAttempted) return;
            _isGpuCheckAttempted = true;

            try
            {
                int result = nvmlInit();
                if (result == 0) // NVML_SUCCESS
                {
                    result = nvmlDeviceGetHandleByIndex(0, out _nvmlDevice);
                    if (result == 0)
                    {
                        _isNvidiaGpu = true;
                        _nvmlInitialized = true;
                        return;
                    }
                }
            }
            catch
            {
                // 失敗時はN/Aにフォールバック
            }

            _isNvidiaGpu = false;
            _nvmlInitialized = false;
        }

        private void UpdateMemoryUsage()
        {
            // 初回実行時にNVMLモニターの初期化を自動で行う（他ファイルの呼び出し順に依存させない）
            if (!_isGpuCheckAttempted)
            {
                InitGpuMonitor();
            }

            // 既存のRAM取得処理
            long workingSet = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64;
            MemoryText.Text = $"RAM: {workingSet / 1024 / 1024} MB";

            // 判定したフラグに応じてタイマー側で高速分岐・取得
            if (_isNvidiaGpu && _nvmlInitialized)
            {
                try
                {
                    if (nvmlDeviceGetUtilizationRates(_nvmlDevice, out var util) == 0 &&
                        nvmlDeviceGetMemoryInfo(_nvmlDevice, out var mem) == 0)
                    {
                        double usedMb = (double)mem.used / (1024 * 1024);
                        double totalMb = (double)mem.total / (1024 * 1024);
                        GpuText.Text = $"GPU: {util.gpu}% (VRAM: {usedMb:F0} / {totalMb:F0} MB)";
                    }
                    else
                    {
                        GpuText.Text = "GPU: Error";
                    }
                }
                catch
                {
                    GpuText.Text = "GPU: N/A";
                }
            }
            else
            {
                // 汎用GPUまたはNVIDIA以外の場合のフォールバック表示
                GpuText.Text = "GPU: N/A (Generic)";
            }
        }
    }
}
