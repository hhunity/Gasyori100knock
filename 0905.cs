
CheckBox toggleButton = new CheckBox();
toggleButton.Appearance = Appearance.Button;   // ボタン風に表示
toggleButton.Text = "OFF";
toggleButton.AutoSize = true;

toggleButton.CheckedChanged += (s, e) =>
{
    toggleButton.Text = toggleButton.Checked ? "ON" : "OFF";
};
this.Controls.Add(toggleButton);

{
  "title": "App Settings",
  "groups": [
    {
      "title": "Camera",
      "items": [
        {
          "key": "exposure",
          "type": "number",
          "min": 0,
          "max": 1000,
          "step": 1,
          "default": 10,
          "unit": "ms",
          "desc": "Exposure time"
        },
        {
          "key": "gain",
          "type": "number",
          "min": 0,
          "max": 24,
          "step": 0.1,
          "default": 0.0,
          "desc": "Analog gain"
        },
        {
          "key": "triggerMode",
          "type": "enum",
          "choices": ["FreeRun", "External"],
          "default": "FreeRun",
          "desc": "Trigger mode"
        }
      ]
    },
    {
      "title": "Processing",
      "items": [
        {
          "key": "enableCuda",
          "type": "bool",
          "default": true,
          "desc": "Use CUDA if available"
        },
        {
          "key": "fftSize",
          "type": "enum",
          "choices": [256, 512, 1024, 2048],
          "default": 512,
          "desc": "FFT window size"
        },
        {
          "key": "outputFolder",
          "type": "string",
          "browse": "folder",
          "default": "C:\\temp",
          "desc": "Output directory"
        },
        {
          "key": "kernelFile",
          "type": "string",
          "browse": "file",
          "filter": "CUDA PTX (*.ptx)|*.ptx|All files (*.*)|*.*",
          "desc": "CUDA kernel file"
        }
      ]
    }
  ]
}


private readonly string _valuesPath =
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                 "MyAppName", "values.json");


using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows.Forms;

public class Form1 : Form
{
    private readonly TabControl _tabs = new TabControl { Dock = DockStyle.Fill };
    private readonly ToolTip _tip = new ToolTip();
    private readonly Dictionary<string, Control> _keyToControl = new(StringComparer.OrdinalIgnoreCase);

    // スキーマ/値ファイルのパス
    private readonly string _schemaPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "schema.json");
    private readonly string _valuesPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "values.json");

    public Form1()
    {
        Text = "Parameter Settings (auto-generated)";
        Width = 900;
        Height = 600;

        var panelBottom = new Panel { Dock = DockStyle.Bottom, Height = 48 };
        var btnLoad = new Button { Text = "Load Values", Width = 120, Left = 10, Top = 10 };
        var btnSave = new Button { Text = "Save Values", Width = 120, Left = 140, Top = 10 };
        var btnReloadSchema = new Button { Text = "Reload Schema", Width = 140, Left = 270, Top = 10 };

        btnLoad.Click += (s, e) => LoadValues(_valuesPath);
        btnSave.Click += (s, e) => SaveValues(_valuesPath);
        btnReloadSchema.Click += (s, e) => ReloadSchema();

        panelBottom.Controls.AddRange(new Control[] { btnLoad, btnSave, btnReloadSchema });

        Controls.Add(_tabs);
        Controls.Add(panelBottom);

        ReloadSchema();
        if (File.Exists(_valuesPath)) LoadValues(_valuesPath);
    }

    private void ReloadSchema()
    {
        _tabs.TabPages.Clear();
        _keyToControl.Clear();

        if (!File.Exists(_schemaPath))
        {
            MessageBox.Show($"Schema not found: {_schemaPath}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        JsonNode root;
        try
        {
            root = JsonNode.Parse(File.ReadAllText(_schemaPath));
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Schema parse error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        Text = GetString(root?["title"]) ?? Text;
        var groups = root?["groups"] as JsonArray;
        if (groups == null)
        {
            MessageBox.Show("Schema has no 'groups' array.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        foreach (var g in groups)
        {
            var gTitle = GetString(g?["title"]) ?? "Group";
            var items = g?["items"] as JsonArray;
            if (items == null) continue;

            var page = new TabPage(gTitle) { Padding = new Padding(12) };

            var table = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 3,
                AutoScroll = true
            };
            table.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));           // Label
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));       // Control
            table.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));           // Unit / Browse

            int row = 0;

            foreach (var item in items)
            {
                string key = GetString(item?["key"]);
                if (string.IsNullOrWhiteSpace(key)) continue;

                string type = (GetString(item?["type"]) ?? "string").ToLowerInvariant();
                string desc = GetString(item?["desc"]) ?? "";
                string unit = GetString(item?["unit"]) ?? "";
                string browse = GetString(item?["browse"])?.ToLowerInvariant();

                // ラベル
                var lbl = new Label { Text = key, AutoSize = true, Anchor = AnchorStyles.Left, Padding = new Padding(0, 6, 6, 0) };

                // エディタ生成
                Control editor = CreateEditor(type, item);

                // ツールチップ
                if (!string.IsNullOrEmpty(desc))
                {
                    _tip.SetToolTip(lbl, desc);
                    _tip.SetToolTip(editor, desc);
                }

                // 右端（Browseボタン or 単位）
                Control right;
                if (!string.IsNullOrEmpty(browse))
                {
                    var btn = new Button { Text = "Browse...", AutoSize = true, Anchor = AnchorStyles.Left };
                    btn.Click += (s, e) =>
                    {
                        if (browse == "folder")
                        {
                            using var dlg = new FolderBrowserDialog();
                            if (dlg.ShowDialog(this) == DialogResult.OK) SetTextLike(editor, dlg.SelectedPath);
                        }
                        else
                        {
                            using var dlg = new OpenFileDialog();
                            string filter = GetString(item?["filter"]) ?? "All files (*.*)|*.*";
                            dlg.Filter = filter;
                            if (dlg.ShowDialog(this) == DialogResult.OK) SetTextLike(editor, dlg.FileName);
                        }
                    };
                    right = btn;
                }
                else if (!string.IsNullOrEmpty(unit))
                {
                    right = new Label { Text = unit, AutoSize = true, Anchor = AnchorStyles.Left, Padding = new Padding(6, 6, 0, 0) };
                }
                else
                {
                    right = new Label { Text = "", AutoSize = true };
                }

                // 行追加
                table.RowStyles.Add(new RowStyle(SizeType.AutoSize));
                table.Controls.Add(lbl, 0, row);
                table.Controls.Add(editor, 1, row);
                table.Controls.Add(right, 2, row);
                row++;

                _keyToControl[key] = editor;
            }

            page.Controls.Add(table);
            _tabs.TabPages.Add(page);
        }

        // スキーマ default 反映
        ApplyDefaults(root);
    }

    private Control CreateEditor(string type, JsonNode item)
    {
        switch (type)
        {
            case "bool":
            {
                var chk = new CheckBox { AutoSize = true, Anchor = AnchorStyles.Left };
                chk.Checked = GetBool(item?["default"]) ?? false;
                return chk;
            }
            case "number":
            {
                var min = GetDecimal(item?["min"], 0m);
                var max = GetDecimal(item?["max"], 1000000m);
                var step = GetDecimal(item?["step"], 1m);

                var num = new NumericUpDown
                {
                    Anchor = AnchorStyles.Left | AnchorStyles.Right,
                    DecimalPlaces = InferDecimalPlacesFromString(GetString(item?["step"])),
                    Minimum = min,
                    Maximum = max,
                    Increment = step,
                    Width = 250
                };
                var defD = GetDouble(item?["default"]);
                if (defD.HasValue)
                {
                    var v = ClampToRange((decimal)defD.Value, num.Minimum, num.Maximum);
                    num.Value = v;
                }
                return num;
            }
            case "enum":
            {
                var combo = new ComboBox
                {
                    DropDownStyle = ComboBoxStyle.DropDownList,
                    Anchor = AnchorStyles.Left | AnchorStyles.Right,
                    Width = 250
                };
                var choices = item?["choices"] as JsonArray;
                if (choices != null)
                {
                    foreach (var c in choices)
                        combo.Items.Add((c as JsonValue)?.ToJsonString().Trim('"') ?? c?.ToString());
                }
                var def = GetString(item?["default"]);
                if (def != null && combo.Items.Contains(def)) combo.SelectedItem = def;
                else if (combo.Items.Count > 0) combo.SelectedIndex = 0;
                return combo;
            }
            case "string":
            default:
            {
                var txt = new TextBox
                {
                    Anchor = AnchorStyles.Left | AnchorStyles.Right,
                    Width = 400
                };
                var def = GetString(item?["default"]);
                if (!string.IsNullOrEmpty(def)) txt.Text = def;
                return txt;
            }
        }
    }

    private void ApplyDefaults(JsonNode root)
    {
        var groups = root?["groups"] as JsonArray;
        if (groups == null) return;

        foreach (var g in groups)
        {
            var items = g?["items"] as JsonArray;
            if (items == null) continue;

            foreach (var item in items)
            {
                string key = GetString(item?["key"]);
                if (string.IsNullOrWhiteSpace(key)) continue;
                if (!_keyToControl.TryGetValue(key, out var ctl)) continue;

                string type = (GetString(item?["type"]) ?? "string").ToLowerInvariant();
                switch (type)
                {
                    case "bool":
                    {
                        bool val = GetBool(item?["default"]) ?? false;
                        if (ctl is CheckBox chk) chk.Checked = val;
                        break;
                    }
                    case "number":
                    {
                        var def = GetDouble(item?["default"]);
                        if (def.HasValue && ctl is NumericUpDown num)
                        {
                            var v = ClampToRange((decimal)def.Value, num.Minimum, num.Maximum);
                            num.Value = v;
                        }
                        break;
                    }
                    case "enum":
                    {
                        string val = GetString(item?["default"]);
                        if (!string.IsNullOrEmpty(val) && ctl is ComboBox cb && cb.Items.Contains(val))
                            cb.SelectedItem = val;
                        break;
                    }
                    default:
                    {
                        string val = GetString(item?["default"]);
                        if (val != null && ctl is TextBox tb) tb.Text = val;
                        break;
                    }
                }
            }
        }
    }

    private void SaveValues(string path)
    {
        var obj = new JsonObject();
        foreach (var kv in _keyToControl)
        {
            string key = kv.Key;
            Control c = kv.Value;
            switch (c)
            {
                case CheckBox chk:
                    obj[key] = chk.Checked;
                    break;
                case NumericUpDown num:
                    obj[key] = (double)num.Value; // 小数も保存
                    break;
                case ComboBox cb:
                    obj[key] = cb.SelectedItem?.ToString();
                    break;
                case TextBox tb:
                    obj[key] = tb.Text;
                    break;
            }
        }

        File.WriteAllText(path, obj.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        MessageBox.Show($"Saved: {path}", "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
    }

    private void LoadValues(string path)
    {
        if (!File.Exists(path))
        {
            MessageBox.Show($"Values not found: {path}", "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }

        JsonNode root;
        try
        {
            root = JsonNode.Parse(File.ReadAllText(path));
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Values parse error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        foreach (var kv in _keyToControl)
        {
            string key = kv.Key;
            var token = root?[key];
            if (token == null) continue;

            var ctl = kv.Value;
            switch (ctl)
            {
                case CheckBox chk:
                    chk.Checked = GetBool(token) ?? chk.Checked;
                    break;

                case NumericUpDown num:
                {
                    decimal v;
                    var d = GetDouble(token);
                    if (d.HasValue) v = (decimal)d.Value;
                    else
                    {
                        var s = GetString(token);
                        if (!decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out v) &&
                            !decimal.TryParse(s, NumberStyles.Any, CultureInfo.CurrentCulture, out v))
                            break;
                    }
                    num.Value = ClampToRange(v, num.Minimum, num.Maximum);
                    break;
                }

                case ComboBox cb:
                {
                    string s = GetString(token);
                    if (s != null && cb.Items.Contains(s)) cb.SelectedItem = s;
                    break;
                }

                case TextBox tb:
                    tb.Text = GetString(token) ?? tb.Text;
                    break;
            }
        }

        MessageBox.Show($"Loaded: {path}", "Info", MessageBoxButtons.OK, MessageBoxIcon.Information);
    }

    // ───────── helpers ─────────

    private static int InferDecimalPlacesFromString(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return 0;
        var dot = s.IndexOf('.');
        return dot >= 0 ? Math.Min(10, s.Length - dot - 1) : 0;
    }

    private static string GetString(JsonNode node)
        => (node as JsonValue)?.ToJsonString().Trim('"') ?? node?.ToString();

    private static bool? GetBool(JsonNode node)
    {
        try { return node?.GetValue<bool>(); }
        catch
        {
            var s = GetString(node);
            if (bool.TryParse(s, out var b)) return b;
            return null;
        }
    }

    private static double? GetDouble(JsonNode node)
    {
        try { return node?.GetValue<double>(); }
        catch
        {
            var s = GetString(node);
            if (double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var d) ||
                double.TryParse(s, NumberStyles.Any, CultureInfo.CurrentCulture, out d))
                return d;
            return null;
        }
    }

    private static decimal GetDecimal(JsonNode node, decimal fallback)
    {
        var d = GetDouble(node);
        if (d.HasValue) return (decimal)d.Value;

        var s = GetString(node);
        if (decimal.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ||
            decimal.TryParse(s, NumberStyles.Any, CultureInfo.CurrentCulture, out v))
            return v;

        return fallback;
    }

    private static decimal ClampToRange(decimal v, decimal min, decimal max)
        => v < min ? min : (v > max ? max : v);

    private static void SetTextLike(Control c, string text)
    {
        switch (c)
        {
            case TextBox tb: tb.Text = text; break;
            case ComboBox cb: cb.Text = text; break;
            default: break;
        }
    }
}