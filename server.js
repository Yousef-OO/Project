const express = require("express");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
app.use(express.json());

// IMPORTANT: مسار predict.py داخل مجلد output
const PYTHON_SCRIPT_PATH = path.join(__dirname, "output", "predict.py");

app.post("/recommend", (req, res) => {
  const input = JSON.stringify(req.body);

  // نشغل Python ونمرّر له JSON للمدخلات
  const python = spawn("python3", [PYTHON_SCRIPT_PATH, input], {
    cwd: __dirname
  });

  let output = "";
  let errorOutput = "";

  python.stdout.on("data", (data) => {
    output += data.toString();
  });

  python.stderr.on("data", (data) => {
    errorOutput += data.toString();
    console.error("PYTHON ERROR:", data.toString());
  });

  python.on("close", (code) => {
    if (errorOutput.trim() !== "") {
      return res.status(500).json({
        error: "Python execution error",
        details: errorOutput
      });
    }

    try {
      const json = JSON.parse(output);
      res.json(json);
    } catch (err) {
      res.status(500).json({
        error: "Invalid JSON from Python",
        output
      });
    }
  });
});

app.listen(3000, () =>
  console.log("🚀 Express server running on http://localhost:3000")
);
