<UserControl x:Class="MyApp.DynamicPropertyEditor"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
    <StackPanel x:Name="FormPanel" />
</UserControl>

using System.Reflection;
using System.Windows.Controls;

public partial class DynamicPropertyEditor : UserControl
{
    private readonly Dictionary<string, Control> inputControls = new();

    public object Target { get; private set; }

    public DynamicPropertyEditor()
    {
        InitializeComponent();
    }

    public void LoadObject<T>(T obj)
    {
        Target = obj!;
        FormPanel.Children.Clear();
        inputControls.Clear();

        var props = typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance);

        foreach (var prop in props)
        {
            var label = new TextBlock { Text = prop.Name, Margin = new Thickness(0, 5, 0, 2) };
            Control input;

            if (prop.PropertyType == typeof(int))
            {
                var tb = new TextBox
                {
                    Text = prop.GetValue(obj)?.ToString(),
                    Width = 100
                };
                input = tb;
            }
            else if (prop.PropertyType == typeof(bool))
            {
                var cb = new CheckBox
                {
                    IsChecked = (bool?)prop.GetValue(obj) ?? false
                };
                input = cb;
            }
            else
            {
                continue; // 未対応型はスキップ
            }

            FormPanel.Children.Add(label);
            FormPanel.Children.Add(input);
            inputControls[prop.Name] = input;
        }
    }

    public T GetEditedObject<T>()
    {
        var newObj = Activator.CreateInstance<T>();
        var props = typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance);

        foreach (var prop in props)
        {
            if (!inputControls.TryGetValue(prop.Name, out var control))
                continue;

            if (control is TextBox tb && int.TryParse(tb.Text, out int i))
                prop.SetValue(newObj, i);
            else if (control is CheckBox cb)
                prop.SetValue(newObj, cb.IsChecked == true);
        }

        return newObj;
    }
}


<Window ... xmlns:local="clr-namespace:MyApp">
    <StackPanel>
        <local:DynamicPropertyEditor x:Name="Editor" />

        <StackPanel Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,10,0,0">
            <Button Content="OK" Width="80" Click="Ok_Click"/>
            <Button Content="キャンセル" Width="80" Click="Cancel_Click"/>
        </StackPanel>
    </StackPanel>
</Window>

private MySettings _original;

public MySettings Result { get; private set; }

public SettingsWindow(MySettings settings)
{
    InitializeComponent();
    _original = settings;
    Editor.LoadObject(settings);
}

#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>

int main() {
    int gpuCount = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA-enabled devices: " << gpuCount << std::endl;

    if (gpuCount > 0) {
        cv::Mat src = cv::imread("test.png", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat d_src, d_blurred;
        d_src.upload(src);

        auto filter = cv::cuda::createGaussianFilter(src.type(), src.type(), cv::Size(5, 5), 1.5);
        filter->apply(d_src, d_blurred);

        cv::Mat result;
        d_blurred.download(result);
        cv::imwrite("blurred.png", result);
    }
}


private void Ok_Click(object sender, RoutedEventArgs e)
{
    Result = Editor.GetEditedObject<MySettings>();
    DialogResult = true;
    Close();
}

var settings = new MySettings { Speed = 5, IsEnabled = true };
var dialog = new SettingsWindow(settings);
if (dialog.ShowDialog() == true)
{
    var updated = dialog.Result;
    // ここで更新された構造体を使用
}







============================================

using CommunityToolkit.Mvvm.ComponentModel;

public partial class SettingsViewModel : ObservableObject
{
    [ObservableProperty]
    private int speed;

    [ObservableProperty]
    private bool isEnabled;

    public SettingsViewModel(MySettings settings)
    {
        Speed = settings.Speed;
        IsEnabled = settings.IsEnabled;
    }

    public MySettings ToStruct()
    {
        return new MySettings
        {
            Speed = this.Speed,
            IsEnabled = this.IsEnabled
        };
    }
}

<Window x:Class="WpfApp1.SettingsWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="設定" Height="200" Width="300">
    <StackPanel Margin="20">
        <TextBlock Text="速度:"/>
        <TextBox Text="{Binding Speed, Mode=TwoWay}" />

        <CheckBox Content="有効" IsChecked="{Binding IsEnabled}" Margin="0,10,0,0"/>

        <StackPanel Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,20,0,0">
            <Button Content="OK" Width="80" Margin="0,0,10,0" Click="Ok_Click"/>
            <Button Content="キャンセル" Width="80" Click="Cancel_Click"/>
        </StackPanel>
    </StackPanel>
</Window>


public partial class SettingsWindow : Window
{
    public SettingsViewModel ViewModel { get; }

    public SettingsWindow(MySettings initial)
    {
        InitializeComponent();
        ViewModel = new SettingsViewModel(initial);
        DataContext = ViewModel;
    }

    private void Ok_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}

呼び出し下
private void OpenSettings_Click(object sender, RoutedEventArgs e)
{
    MySettings current = new MySettings { Speed = 10, IsEnabled = true };

    var dialog = new SettingsWindow(current);
    if (dialog.ShowDialog() == true)
    {
        MySettings updated = dialog.ViewModel.ToStruct();
        MessageBox.Show($"Speed: {updated.Speed}, IsEnabled: {updated.IsEnabled}");
    }
}








==========================================


using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace MyControls
{
    public class NumericTextBox : TextBox
    {
        public static readonly DependencyProperty ValueProperty =
            DependencyProperty.Register(nameof(Value), typeof(int), typeof(NumericTextBox),
                new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault, OnValueChanged));

        public int Value
        {
            get => (int)GetValue(ValueProperty);
            set => SetValue(ValueProperty, value);
        }

        static NumericTextBox()
        {
            // スタイル適用を有効にする場合必要
            DefaultStyleKeyProperty.OverrideMetadata(typeof(NumericTextBox), 
                new FrameworkPropertyMetadata(typeof(NumericTextBox)));
        }

        public NumericTextBox()
        {
            Text = Value.ToString();
        }

        protected override void OnPreviewMouseWheel(MouseWheelEventArgs e)
        {
            int delta = e.Delta > 0 ? 1 : -1;
            Value += delta;
            e.Handled = true;
        }

        protected override void OnTextChanged(TextChangedEventArgs e)
        {
            if (int.TryParse(Text, out int parsed))
            {
                Value = parsed;
            }
            base.OnTextChanged(e);
        }

        private static void OnValueChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var control = (NumericTextBox)d;
            if (control.Text != control.Value.ToString())
            {
                control.Text = control.Value.ToString();
            }
        }
    }
}

<Window x:Class="WpfApp1.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:local="clr-namespace:WpfApp1"
        xmlns:ctrl="clr-namespace:MyControls"
        Title="MainWindow" Height="200" Width="300">

    <Window.DataContext>
        <local:MainViewModel />
    </Window.DataContext>

    <Grid>
        <ctrl:NumericTextBox Value="{Binding Speed, Mode=TwoWay}"
                              HorizontalAlignment="Center"
                              VerticalAlignment="Center"
                              Width="100" FontSize="20" TextAlignment="Center"/>
    </Grid>
</Window>



using CommunityToolkit.Mvvm.ComponentModel;

public partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    private int speed = 10;
}






