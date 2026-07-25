using System.Windows;

namespace ZiView;

/// <summary>
/// TensorRTエンジン構築・ウォームアップ等、UIスレッドをブロックしうる重い初期化処理の間、
/// 「処理中である」ことをユーザーに示すための非モーダルなオーバーレイ表示。
/// ボタンは持たず、呼び出し側がShow/Closeを制御する（AppMessageBoxDialogとは別系統の軽量ウィンドウ）。
/// </summary>
public partial class EngineStatusOverlay : Window
{
    public EngineStatusOverlay(string message)
    {
        InitializeComponent();
        TbMessage.Text = message;
    }

    public void SetMessage(string message) => TbMessage.Text = message;
}
