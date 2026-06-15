// LensCorrectionEffect.hlsl
sampler2D implicitSampler : register(S0);
float distortionAmount : register(C0); // プラスで樽型補正、マイナスで糸巻き型補正

float4 main(float2 uv : TEXCOORD) : COLOR
{
    // 中心座標を (0,0) として基準化
    float2 cc = uv - 0.5;
    float dist = dot(cc, cc);
    
    // 符号とサンプリング方向のベクトルを修正
    // 物理湾曲（画面が引っ込んでいる、または飛び出ている）を打ち消すための逆算
    float2 targetUV = uv - cc * dist * distortionAmount;
    
    // テクスチャの境界チェック
    if (targetUV.x >= 0.0 && targetUV.x <= 1.0 && targetUV.y >= 0.0 && targetUV.y <= 1.0)
    {
        return tex2D(implicitSampler, targetUV);
    }
    else
    {
        // 補正によって削られた外側は黒（背景に馴染むように）
        return float4(0.0, 0.0, 0.0, 1.0);
    }
}