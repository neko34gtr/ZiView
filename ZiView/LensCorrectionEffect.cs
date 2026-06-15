using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Effects;

namespace ZiView
{
    /// <summary>
    /// 逆湾曲（レンズクロス・歪み）をリアルタイム補正するカスタムシェーダーエフェクト
    /// </summary>
    public class LensCorrectionEffect : ShaderEffect
    {
        public static readonly DependencyProperty InputProperty =
            RegisterPixelShaderSamplerProperty("Input", typeof(LensCorrectionEffect), 0);

        public static readonly DependencyProperty DistortionAmountProperty =
            DependencyProperty.Register("DistortionAmount", typeof(double), typeof(LensCorrectionEffect),
                new UIPropertyMetadata(0.0, PixelShaderConstantCallback(0)));

        public LensCorrectionEffect()
        {
            PixelShader shader = new PixelShader();
            // 事前にコンパイルされたHLSLのピクセルシェーダーバイナリ（.ps）へのリソース定義
            shader.UriSource = new Uri("pack://application:,,,/ZiView;component/LensCorrectionEffect.ps", UriKind.Absolute);
            this.PixelShader = shader;

            // 依存プロパティとシェーダーレジスタをバインド
            UpdateShaderValue(InputProperty);
            UpdateShaderValue(DistortionAmountProperty);
        }

        /// <summary>
        /// サンプラーソース画像
        /// </summary>
        public Brush Input
        {
            get { return ((Brush)(this.GetValue(InputProperty))); }
            set { this.SetValue(InputProperty, value); }
        }

        /// <summary>
        /// 歪み補正係数 (K値)
        /// </summary>
        public double DistortionAmount
        {
            get { return ((double)(this.GetValue(DistortionAmountProperty))); }
            set { this.SetValue(DistortionAmountProperty, value); }
        }
    }
}