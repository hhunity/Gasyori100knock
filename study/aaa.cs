using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace FirmwareWizard;

public class StepIndexToVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is int currentStep && parameter is string targetStepStr && int.TryParse(targetStepStr, out var targetStep))
            return currentStep == targetStep ? Visibility.Visible : Visibility.Collapsed;

        return Visibility.Collapsed;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.IO.Ports;
using System.Windows.Media;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;

namespace FirmwareWizard;

public partial class MainViewModel : ObservableObject
{
    private readonly string _avrdudePath = Path.Combine(AppContext.BaseDirectory, "tools", "avrdude", "avrdude.exe");
    private readonly string _avrdudeConfPath = Path.Combine(AppContext.BaseDirectory, "tools", "avrdude", "avrdude.conf");

    public MainViewModel()
    {
        RefreshPorts();
    }

    // ==================== ステップ管理 ====================

    [ObservableProperty]
    private int currentStep;

    partial void OnCurrentStepChanged(int value)
    {
        OnPropertyChanged(nameof(TitleText));
        OnPropertyChanged(nameof(NextButtonText));
        OnPropertyChanged(nameof(IsBackVisible));
        NextCommand.NotifyCanExecuteChanged();
        BackCommand.NotifyCanExecuteChanged();
    }

    public string TitleText => CurrentStep switch
    {
        0 => "ステップ 1 / 5: COMポート選択",
        1 => "ステップ 2 / 5: ファーム選択",
        2 => "ステップ 3 / 5: 確認",
        3 => "ステップ 4 / 5: 書き込み中",
        4 => "ステップ 5 / 5: 完了",
        _ => ""
    };

    public string NextButtonText => CurrentStep switch
    {
        2 => "書き込み開始",
        4 => "閉じる",
        _ => "次へ"
    };

    public bool IsBackVisible => CurrentStep != 4;

    // ==================== Step 1: COMポート(差分検出) ====================

    public ObservableCollection<string> ComPorts { get; } = new();

    [ObservableProperty]
    private string? selectedPort;

    partial void OnSelectedPortChanged(string? value) => NextCommand.NotifyCanExecuteChanged();

    public ObservableCollection<string> BaudRates { get; } = new() { "115200", "57600", "9600" };

    [ObservableProperty]
    private string selectedBaudRate = "115200";

    [ObservableProperty]
    private string detectStatusText = "";

    [ObservableProperty]
    private Brush detectStatusColor = Brushes.Gray;

    [ObservableProperty]
    private bool isDetecting;

    partial void OnIsDetectingChanged(bool value)
    {
        AutoDetectCommand.NotifyCanExecuteChanged();
        NextCommand.NotifyCanExecuteChanged();
    }

    private DispatcherTimer? _detectTimer;
    private HashSet<string> _portsBeforeDetect = new();
    private int _detectElapsedTicks;
    private const int DetectTimeoutSeconds = 15;

    [RelayCommand(CanExecute = nameof(CanAutoDetect))]
    private void AutoDetect()
    {
        _portsBeforeDetect = SerialPort.GetPortNames().ToHashSet();
        _detectElapsedTicks = 0;

        IsDetecting = true;
        DetectStatusText = "USBケーブルを接続してください...";
        DetectStatusColor = Brushes.DarkOrange;

        _detectTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        _detectTimer.Tick += DetectTimer_Tick;
        _detectTimer.Start();
    }

    private bool CanAutoDetect() => !IsDetecting;

    private void DetectTimer_Tick(object? sender, EventArgs e)
    {
        var currentPorts = SerialPort.GetPortNames().ToHashSet();
        var newPorts = currentPorts.Except(_portsBeforeDetect).ToList();

        if (newPorts.Count > 0)
        {
            StopDetectTimer();

            var detectedPort = newPorts.First();
            RefreshPorts();
            SelectedPort = detectedPort;

            DetectStatusText = $"検出しました: {detectedPort}";
            DetectStatusColor = Brushes.SeaGreen;
            IsDetecting = false;
            return;
        }

        _detectElapsedTicks++;
        if (_detectElapsedTicks * 0.5 >= DetectTimeoutSeconds)
        {
            StopDetectTimer();
            DetectStatusText = "検出できませんでした。手動でポートを選択してください。";
            DetectStatusColor = Brushes.Firebrick;
            IsDetecting = false;
        }
    }

    private void StopDetectTimer()
    {
        _detectTimer?.Stop();
        _detectTimer = null;
    }

    [RelayCommand]
    private void RefreshPorts()
    {
        var selected = SelectedPort;
        ComPorts.Clear();
        foreach (var p in SerialPort.GetPortNames().OrderBy(p => p))
            ComPorts.Add(p);

        if (selected != null && ComPorts.Contains(selected))
            SelectedPort = selected;
        else if (ComPorts.Count > 0)
            SelectedPort = ComPorts[0];
    }

    // ==================== Step 2: HEXファイル ====================

    [ObservableProperty]
    private string hexPath = "";

    partial void OnHexPathChanged(string value) => NextCommand.NotifyCanExecuteChanged();

    [RelayCommand]
    private void BrowseHex()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "HEXファイル (*.hex)|*.hex|すべてのファイル (*.*)|*.*",
            Title = "ファームウェアファイルを選択"
        };

        if (dialog.ShowDialog() == true)
            HexPath = dialog.FileName;
    }

    // ==================== Step 4: 書き込み ====================

    [ObservableProperty]
    private string logText = "";

    [ObservableProperty]
    private bool isFlashing;

    [ObservableProperty]
    private string resultText = "";

    [ObservableProperty]
    private Brush resultColor = Brushes.Black;

    [ObservableProperty]
    private string resultDetailText = "";

    private void AppendLog(string line) => LogText += line + Environment.NewLine;

    private Task RunAvrdudeAsync()
    {
        IsFlashing = true;
        var comPort = SelectedPort ?? "";
        var baud = SelectedBaudRate;
        var hexPathLocal = HexPath;

        LogText = "";
        AppendLog($"avrdude を実行します: ポート={comPort}, ボーレート={baud}");
        AppendLog($"HEXファイル: {hexPathLocal}");
        AppendLog("----------------------------------------");

        var tcs = new TaskCompletionSource();

        var psi = new ProcessStartInfo
        {
            FileName = _avrdudePath,
            Arguments = $"-c wiring -p atmega2560 -P {comPort} -b {baud} -D -U flash:w:\"{hexPathLocal}\":i -C \"{_avrdudeConfPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        var process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        process.OutputDataReceived += (s, e) => { if (e.Data != null) AppendLog(e.Data); };
        process.ErrorDataReceived += (s, e) => { if (e.Data != null) AppendLog(e.Data); };

        process.Exited += (s, e) =>
        {
            var succeeded = process.ExitCode == 0;
            IsFlashing = false;

            AppendLog("----------------------------------------");
            AppendLog(succeeded ? "書き込みが正常に完了しました。" : $"書き込みに失敗しました。(終了コード: {process.ExitCode})");

            ResultText = succeeded ? "✅ 書き込みが完了しました" : "❌ 書き込みに失敗しました";
            ResultColor = succeeded ? Brushes.SeaGreen : Brushes.Firebrick;
            ResultDetailText = succeeded
                ? "Arduino Megaへのファームウェア更新が完了しました。"
                : "ログを確認し、COMポート・ケーブル接続・HEXファイルを確認してください。";

            tcs.TrySetResult();
        };

        try
        {
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            IsFlashing = false;
            AppendLog($"avrdudeの起動に失敗しました: {ex.Message}");
            AppendLog($"確認したパス: {_avrdudePath}");

            ResultText = "❌ avrdudeの起動に失敗しました";
            ResultColor = Brushes.Firebrick;
            ResultDetailText = "avrdude.exeの配置パスが正しいか確認してください。";

            tcs.TrySetResult();
        }

        return tcs.Task;
    }

    // ==================== 次へ/戻る ====================

    [RelayCommand(CanExecute = nameof(CanGoNext))]
    private async Task Next()
    {
        switch (CurrentStep)
        {
            case 0:
                CurrentStep = 1;
                break;

            case 1:
                CurrentStep = 2;
                break;

            case 2:
                CurrentStep = 3;
                await RunAvrdudeAsync();
                CurrentStep = 4;
                break;

            case 4:
                System.Windows.Application.Current.Shutdown();
                break;
        }
    }

    private bool CanGoNext()
    {
        if (IsFlashing) return false;

        return CurrentStep switch
        {
            0 => !string.IsNullOrEmpty(SelectedPort),
            1 => !string.IsNullOrEmpty(HexPath) && File.Exists(HexPath),
            3 => false,
            _ => true
        };
    }

    [RelayCommand(CanExecute = nameof(CanGoBack))]
    private void Back()
    {
        if (CurrentStep > 0) CurrentStep--;
    }

    private bool CanGoBack() => CurrentStep > 0 && CurrentStep != 3 && !IsFlashing;
}

<Window x:Class="FirmwareWizard.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:local="clr-namespace:FirmwareWizard"
        Title="ファームウェア更新ウィザード"
        Height="440" Width="560"
        WindowStartupLocation="CenterScreen"
        ResizeMode="NoResize">

    <Window.DataContext>
        <local:MainViewModel/>
    </Window.DataContext>

 <Window.Resources>
    <local:StepIndexToVisibilityConverter x:Key="StepConverter"/>
    <BooleanToVisibilityConverter x:Key="BoolToVis"/>
</Window.Resources>


    <Grid Margin="20">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <TextBlock Grid.Row="0" Text="{Binding TitleText}"
                   FontSize="18" FontWeight="Bold" Margin="0,0,0,15"/>

        <Grid Grid.Row="1">

            <!-- Step 1: COMポート選択(差分検出方式) -->
            <StackPanel Visibility="{Binding CurrentStep, Converter={StaticResource StepConverter}, ConverterParameter=0}">
                <TextBlock Text="Arduino Megaをまだ接続していない状態で「自動検出」を押し、" Margin="0,0,0,2"/>
                <TextBlock Text="指示が出たらUSBケーブルを差し込んでください。" Margin="0,0,0,10"/>

                <StackPanel Orientation="Horizontal" Margin="0,0,0,10">
                    <Button Content="自動検出" Width="100" Command="{Binding AutoDetectCommand}"/>
                    <TextBlock Text="{Binding DetectStatusText}" Foreground="{Binding DetectStatusColor}"
                               VerticalAlignment="Center" Margin="10,0,0,0"/>
                </StackPanel>

                <StackPanel Orientation="Horizontal" Margin="0,0,0,10">
                    <TextBlock Text="検出結果 / 手動選択: " VerticalAlignment="Center" Margin="0,0,10,0"/>
                    <ComboBox Width="150" Margin="0,0,10,0"
                              ItemsSource="{Binding ComPorts}" SelectedItem="{Binding SelectedPort}"/>
                    <Button Content="一覧更新" Width="80" Command="{Binding RefreshPortsCommand}"/>
                </StackPanel>

                <StackPanel Orientation="Horizontal">
                    <TextBlock Text="ボーレート: " VerticalAlignment="Center" Margin="0,0,10,0"/>
                    <ComboBox Width="120" ItemsSource="{Binding BaudRates}" SelectedItem="{Binding SelectedBaudRate}"/>
                </StackPanel>
            </StackPanel>

            <!-- Step 2: HEXファイル選択 -->
            <StackPanel Visibility="{Binding CurrentStep, Converter={StaticResource StepConverter}, ConverterParameter=1}">
                <TextBlock Text="書き込むファームウェア(.hex)ファイルを選択してください。" Margin="0,0,0,10"/>
                <StackPanel Orientation="Horizontal">
                    <TextBox Width="330" Margin="0,0,10,0" IsReadOnly="True" Text="{Binding HexPath, Mode=OneWay}"/>
                    <Button Content="参照..." Width="80" Command="{Binding BrowseHexCommand}"/>
                </StackPanel>
            </StackPanel>

            <!-- Step 3: 確認画面 -->
            <StackPanel Visibility="{Binding CurrentStep, Converter={StaticResource StepConverter}, ConverterParameter=2}">
                <TextBlock Text="以下の内容で書き込みを行います。よろしいですか?" Margin="0,0,0,15" FontWeight="Bold"/>
                <TextBlock Text="COMポート:" FontWeight="SemiBold"/>
                <TextBlock Text="{Binding SelectedPort}" Margin="0,0,0,10"/>
                <TextBlock Text="ボーレート:" FontWeight="SemiBold"/>
                <TextBlock Text="{Binding SelectedBaudRate}" Margin="0,0,0,10"/>
                <TextBlock Text="HEXファイル:" FontWeight="SemiBold"/>
                <TextBlock Text="{Binding HexPath}" TextWrapping="Wrap"/>
            </StackPanel>

            <!-- Step 4: 書き込み中 -->
            <StackPanel Visibility="{Binding CurrentStep, Converter={StaticResource StepConverter}, ConverterParameter=3}">
                <TextBlock Text="書き込み中です。しばらくお待ちください..." Margin="0,0,0,10"/>
                <ProgressBar Height="20" IsIndeterminate="True" Margin="0,0,0,10"/>
                <TextBox Height="200" IsReadOnly="True" Text="{Binding LogText, Mode=OneWay}"
                         VerticalScrollBarVisibility="Auto" TextWrapping="NoWrap"
                         FontFamily="Consolas" FontSize="11" Background="#1E1E1E" Foreground="#DCDCDC"/>
            </StackPanel>

            <!-- Step 5: 完了 -->
            <StackPanel Visibility="{Binding CurrentStep, Converter={StaticResource StepConverter}, ConverterParameter=4}">
                <TextBlock Text="{Binding ResultText}" Foreground="{Binding ResultColor}"
                           FontSize="16" FontWeight="Bold" Margin="0,20,0,10"/>
                <TextBlock Text="{Binding ResultDetailText}" TextWrapping="Wrap"/>
            </StackPanel>

        </Grid>

        <StackPanel Grid.Row="2" Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,15,0,0">
            <Button Content="戻る" Width="90" Height="30" Margin="0,0,10,0"
                    Command="{Binding BackCommand}" Visibility="{Binding IsBackVisible, Converter={StaticResource BoolToVis}}"/>
            <Button Content="{Binding NextButtonText}" Width="90" Height="30" Command="{Binding NextCommand}"/>
        </StackPanel>

    </Grid>
</Window>




