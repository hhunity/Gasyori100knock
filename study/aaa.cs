public partial class InspectionViewModel : ObservableObject
{
    public ObservableCollection<TestStep> Steps { get; } = new();

    [ObservableProperty]
    private bool isRunning;

    public IAsyncRelayCommand RunAllCommand { get; }

    public InspectionViewModel()
    {
        // あらかじめ全項目を並べておく
        Steps.Add(new TestStep("電源電圧チェック"));
        Steps.Add(new TestStep("通信確認"));
        Steps.Add(new TestStep("外観検査"));
        Steps.Add(new TestStep("動作確認"));

        RunAllCommand = new AsyncRelayCommand(RunAllAsync, () => !IsRunning);
    }

    private async Task RunAllAsync()
    {
        IsRunning = true;

        foreach (var step in Steps)
        {
            step.Status = TestStatus.Running;
            bool ok = await SimulateTestAsync();
            step.Status = ok ? TestStatus.Pass : TestStatus.Fail;

            if (!ok) break;
        }

        IsRunning = false;
    }

    private async Task<bool> SimulateTestAsync()
    {
        await Task.Delay(500);
        return true;
    }
}

public partial class TestStep : ObservableObject
{
    public string Name { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatusText))]
    private TestStatus status = TestStatus.Pending;

    public string StatusText => Status switch
    {
        TestStatus.Pending => "-",
        TestStatus.Running => "実行中",
        TestStatus.Pass => "PASS",
        TestStatus.Fail => "FAIL",
        _ => ""
    };

    public TestStep(string name) => Name = name;
}


<Window.Resources>
    <local:StatusToBrushConverter x:Key="StatusToBrush"/>
</Window.Resources>

<StackPanel>
    <Button Content="検査開始" Command="{Binding RunAllCommand}"
            Width="120" Height="40" Margin="10"/>

    <DataGrid ItemsSource="{Binding Steps}"
              AutoGenerateColumns="False"
              IsReadOnly="True"
              HeadersVisibility="Column"
              CanUserAddRows="False"
              GridLinesVisibility="Horizontal"
              RowHeight="40">
        <DataGrid.Columns>
            <DataGridTextColumn Header="項目" Binding="{Binding Name}" Width="*"/>
            <DataGridTemplateColumn Header="結果" Width="120">
                <DataGridTemplateColumn.CellTemplate>
                    <DataTemplate>
                        <TextBlock Text="{Binding StatusText}"
                                   FontWeight="Bold"
                                   FontSize="16"
                                   HorizontalAlignment="Center"
                                   VerticalAlignment="Center"
                                   Foreground="{Binding Status, Converter={StaticResource StatusToBrush}}"/>
                    </DataTemplate>
                </DataGridTemplateColumn.CellTemplate>
            </DataGridTemplateColumn>
        </DataGrid.Columns>
    </DataGrid>
</StackPanel>

