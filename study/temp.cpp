// 1. グラフのインスタンスを作成
const myChart = new Chart(ctx, config);

// 2. 保存ボタンなどのイベントで実行
function saveChart() {
    const imageLink = document.createElement('a');
    const canvas = document.getElementById('myChartCanvas');
    
    // 背景を白くしたい場合は、一度Canvasの背景を塗る設定が必要です（後述）
    imageLink.download = 'my-chart.png';
    imageLink.href = myChart.toBase64Image(); // Base64文字列を取得
    
    imageLink.click();
}
