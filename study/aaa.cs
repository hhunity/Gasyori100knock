{
  "pinNames": {
    "LED_BUILTIN": 13,
    "RELAY_1": 9,
    "SENSOR_1": 7,
    "SENSOR_2": 8
  },
  "testCases": [
    {
      "name": "TC001_Relay_Check",
      "description": "リレーON後、100ms待ってから状態確認",
      "steps": [
        { "action": "reset" },
        { "action": "setMode", "pin": "RELAY_1",  "mode": "OUTPUT" },
        { "action": "write",   "pin": "RELAY_1",  "value": "HIGH" },
        { "action": "delay",   "ms": 100 },
        { "action": "setMode", "pin": "SENSOR_1", "mode": "INPUT" },
        { "action": "read",    "pin": "SENSOR_1", "expected": "HIGH" }
      ]
    },
    {
      "name": "TC002_LED_Off_Check",
      "description": "LED消灯確認",
      "steps": [
        { "action": "reset" },
        { "action": "setMode", "pin": "LED_BUILTIN", "mode": "OUTPUT" },
        { "action": "write",   "pin": "LED_BUILTIN", "value": "LOW" },
        { "action": "setMode", "pin": "SENSOR_2",    "mode": "INPUT" },
        { "action": "read",    "pin": "SENSOR_2",    "expected": "LOW" }
      ]
    }
  ]
}

using System;
using System.IO;
using System.IO.Ports;
using System.Text.Json;
using System.Collections.Generic;
using System.Threading;

public class Step
{
    public string Action { get; set; }
    public string Pin { get; set; }
    public string Mode { get; set; }
    public string Value { get; set; }
    public string Expected { get; set; }
    public int Ms { get; set; }
}

public class TestCase
{
    public string Name { get; set; }
    public string Description { get; set; }
    public List<Step> Steps { get; set; }
}

public class TestCaseFile
{
    public Dictionary<string, int> PinNames { get; set; }
    public List<TestCase> TestCases { get; set; }
}

class Program
{
    static SerialPort port;
    static Dictionary<string, int> pinMap;

    static void Main()
    {
        string json = File.ReadAllText("testcases.json");
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var data = JsonSerializer.Deserialize<TestCaseFile>(json, options);

        pinMap = data.PinNames;

        port = new SerialPort("COM3", 9600);
        port.NewLine = "\n";
        port.ReadTimeout = 2000;
        port.Open();
        Thread.Sleep(2000);
        port.ReadLine();  // "Ready" 読み捨て

        foreach (var tc in data.TestCases)
        {
            RunTestCase(tc);
        }

        port.Close();
    }

    static void RunTestCase(TestCase tc)
    {
        Console.WriteLine($"=== {tc.Name}: {tc.Description} ===");
        bool pass = true;

        foreach (var step in tc.Steps)
        {
            // --- delay ---
            if (step.Action == "delay")
            {
                Console.WriteLine($"  Waiting {step.Ms}ms...");
                Thread.Sleep(step.Ms);
                continue;
            }

            // --- reset ---
            if (step.Action == "reset")
            {
                ResetAllPins();
                continue;
            }

            // --- 通常のピン操作(setMode/write/read) ---
            if (!pinMap.TryGetValue(step.Pin, out int pinNumber))
            {
                Console.WriteLine($"  ERR: unknown pin name '{step.Pin}'");
                pass = false;
                continue;
            }

            string result = RunStep(step, pinNumber);

            if (step.Action == "read" && step.Expected != null)
            {
                bool ok = (result == step.Expected);
                Console.WriteLine($"  {step.Pin}(D{pinNumber}): got={result}, expected={step.Expected} → {(ok ? "PASS" : "FAIL")}");
                if (!ok) pass = false;
            }
            else
            {
                Console.WriteLine($"  {step.Action} {step.Pin}(D{pinNumber}): {result}");
            }
        }

        Console.WriteLine(pass ? $"{tc.Name}: PASS\n" : $"{tc.Name}: FAIL\n");
    }

    static string RunStep(Step step, int pinNumber)
    {
        string cmd = step.Action switch
        {
            "setMode" => $"PINMODE {pinNumber} {step.Mode}",
            "write"   => $"WRITE {pinNumber} {step.Value}",
            "read"    => $"READ {pinNumber}",
            _ => throw new Exception($"unknown action: {step.Action}")
        };

        port.WriteLine(cmd);
        return port.ReadLine().Trim();
    }

    static void ResetAllPins()
    {
        foreach (var kv in pinMap)
        {
            string pinName = kv.Key;
            int pinNumber = kv.Value;

            port.WriteLine($"PINMODE {pinNumber} OUTPUT");
            port.ReadLine();
            port.WriteLine($"WRITE {pinNumber} LOW");
            port.ReadLine();
        }
        Console.WriteLine("  All pins reset to LOW");
    }
}


void setup() {
  Serial.begin(9600);
  Serial.println("Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    handleCommand(line);
  }
}

void handleCommand(String line) {
  int sp1 = line.indexOf(' ');
  String cmd = line.substring(0, sp1);
  String rest = line.substring(sp1 + 1);

  if (cmd == "PINMODE") {
    int sp2 = rest.indexOf(' ');
    int pin = rest.substring(0, sp2).toInt();
    String mode = rest.substring(sp2 + 1);

    if (mode == "OUTPUT") pinMode(pin, OUTPUT);
    else if (mode == "INPUT") pinMode(pin, INPUT);
    else if (mode == "INPUT_PULLUP") pinMode(pin, INPUT_PULLUP);

    Serial.println("OK");
  }
  else if (cmd == "WRITE") {
    int sp2 = rest.indexOf(' ');
    int pin = rest.substring(0, sp2).toInt();
    String value = rest.substring(sp2 + 1);

    digitalWrite(pin, value == "HIGH" ? HIGH : LOW);
    Serial.println("OK");
  }
  else if (cmd == "READ") {
    int pin = rest.toInt();
    int value = digitalRead(pin);
    Serial.println(value == HIGH ? "HIGH" : "LOW");
  }
  else {
    Serial.println("ERR: unknown command");
  }
}


