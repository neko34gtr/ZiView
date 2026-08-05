namespace ZiView
{
    /// <summary>
    /// 見開き表示の開き方向。
    /// ・RightOpen（R2）: 従来のマンガ方式。現在ページ(index)を右、次ページ(index+1)を左に配置する右開き2ページ
    /// ・LeftOpen（L2）: 洋書方式。現在ページ(index)を左、次ページ(index+1)を右に配置する左開き2ページ
    /// ・Single（S1）: 見開きなし・単ページ表示
    /// </summary>
    public enum PageOpenMode
    {
        LeftOpen,
        Single,
        RightOpen
    }
}
