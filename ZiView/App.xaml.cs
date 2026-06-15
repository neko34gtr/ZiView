using System;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace ZiView
{
    /// <summary>
    /// アプリケーション全体のライフサイクル、二重起動制御、およびプロセス間通信(IPC)を統括するクラス。
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            // 二重起動チェック等のエントリーポイントとして機能させるため、
            // ここでの直接的なMainWindow生成・読み込みは行いません。
        }
        // OSレベルでの二重起動チェックを行うためのミューテックスインスタンス
        private static Mutex? _mutex;
        // ミューテックスのシステム一意識別名（Globalプレフィックスにより全セッションで一意化）
        private const string MutexName = "Global\\ZiView_Unique_Mutex_String";
        // 後続プロセスから画像パスを受け取るための名前付きパイプ識別名
        private const string PipeName = "ZiView_Pipe";
        // パイプサーバーの非同期待受タスクを安全に破棄するためのトークンソース
        private CancellationTokenSource? _cts;

        /// <summary>
        /// アプリケーション起動時の初期化処理。二重起動の判定と、それに応じた分岐を行います。
        /// </summary>
        private async void Application_Startup(object sender, StartupEventArgs e)
        {
            // ミューテックスを生成し、所有権の取得を試みる
            _mutex = new Mutex(true, MutexName, out bool isNewInstance);

            if (!isNewInstance)
            {
                // すでに他インスタンスが起動している場合（後続プロセス）
                if (e.Args.Length > 0)
                {
                    // コマンドライン引数（画像パス）がある場合は、先行起動中のプロセスへパイプ経由で送信
                    await SendArgsToFirstInstanceAsync(e.Args[0]);
                }
                // ミューテックスを解放し、即座にアプリケーションを完全終了（リソースリーク防止）
                _mutex.Dispose();
                Current.Shutdown();
                return;
            }

            // 自身が最初のインスタンスである場合、後続プロセスからのファイルパスを受け付けるサーバーを起動
            StartPipeServer();

            // メインウィンドウを生成・表示
            var mainWindow = new MainWindow();
            mainWindow.Show();

            // 自身の起動時引数にファイルパスが含まれていれば、初期画像としてロード
            if (e.Args.Length > 0)
            {
                mainWindow.LoadImage(e.Args[0]);
            }
        }

        /// <summary>
        /// 別インスタンスから送られてくるファイルパスを常時監視・受信する非同期パイプサーバー。
        /// </summary>
        private void StartPipeServer()
        {
            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            // UIスレッドをロックしないよう、完全にバックグラウンドタスクとして運用
            Task.Run(async () =>
            {
                while (!token.IsCancellationRequested)
                {
                    try
                    {
                        // 非同期、単一方向の受信専用パイプストリームを生成
                        using var server = new NamedPipeServerStream(PipeName, PipeDirection.In, 1, PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
                        // 後続プロセスからの接続要求を非同期で待機
                        await server.WaitForConnectionAsync(token);

                        // 受信データをUTF-8ストリームとして読み込み
                        using var reader = new StreamReader(server, Encoding.UTF8);
                        string? filePath = await reader.ReadLineAsync(token);

                        if (!string.IsNullOrEmpty(filePath))
                        {
                            // メインウィンドウの操作を行うため、UIスレッド（Dispatcher）に処理を委譲
                            await Current.Dispatcher.InvokeAsync(() =>
                            {
                                if (Current.MainWindow is MainWindow mainWin)
                                {
                                    // 送られてきた画像を読み込んで最前面化
                                    mainWin.LoadImage(filePath);
                                    if (mainWin.WindowState == WindowState.Minimized)
                                    {
                                        mainWin.WindowState = WindowState.Normal;
                                    }
                                    mainWin.Activate();
                                }
                            });
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        // タスクキャンセル時は正常ルートとして例外を無視
                    }
                    catch (Exception ex)
                    {
                        // 不測のエラーはデバッグ出力へ記録し、サーバーのハングアップを回避してループを継続
                        System.Diagnostics.Debug.WriteLine($"Pipe server error: {ex.Message}");
                    }
                }
            }, token);
        }

        /// <summary>
        /// 後続プロセスとして起動した場合に、先行プロセスへファイルパスを転送するクライアント処理。
        /// </summary>
        private async Task SendArgsToFirstInstanceAsync(string filePath)
        {
            try
            {
                // ローカルマシン上のパイプサーバーに対して送信専用モードで接続
                using var client = new NamedPipeClientStream(".", PipeName, PipeDirection.Out);
                // 接続タイムアウトを2秒に設定（デッドロック防止）
                await client.ConnectAsync(2000);

                // パス文字列をUTF-8で安全に書き込み
                using var writer = new StreamWriter(client, Encoding.UTF8);
                await writer.WriteLineAsync(filePath);
                await writer.FlushAsync();
            }
            catch (Exception ex)
            {
                // 通信失敗時は、ユーザーに検知させるため警告ポップアップを表示
                MessageBox.Show($"先行プロセスへの通信に失敗しました:\n{ex.Message}", "ZiView Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// アプリケーション終了時のクリーンアップ。OSリソースを確実に解放します。
        /// </summary>
        private void Application_Exit(object sender, ExitEventArgs e)
        {
            // バックグラウンドのパイプ待受タスクを即座に破棄
            _cts?.Cancel();

            // ミューテックスの所有権を安全に手放し、ハンドルをクローズ
            if (_mutex != null)
            {
                try
                {
                    _mutex.ReleaseMutex();
                }
                catch (ObjectDisposedException) { }
                catch (ApplicationException) { } // 所有権を保持していない場合の例外をケア

                _mutex.Dispose();
            }
        }
    }
}