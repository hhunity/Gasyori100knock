検知の覚える方式戻す。
縦幅制限やっぱり必要？
進捗バーおそい
メモリリーク確認
スループット、レイテンシーの数字出せるように
テキストエディタ選択
テストまわす





#pragma pack(push, 1)   // 1バイト境界に詰める
struct A {
    char c;     // 1 byte
    int  i;     // 4 byte
};
#pragma pack(pop)

[StructLayout(LayoutKind.Sequential, Pack = 1)]
struct A {
    public byte c;
    public int i;
}

