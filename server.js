const express = require("express");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
app.use(express.json());

// predicted.py path
const PYTHON_SCRIPT_PATH = path.join(__dirname, "output", "predict.py");

// serve front.html directly
app.use(express.static(__dirname));

app.post("/recommend", (req, res) => {
  const input = JSON.stringify(req.body);

  const python = spawn("python", [PYTHON_SCRIPT_PATH, input], {
    cwd: __dirname,
  });

  let output = "";
  let errorOutput = "";

  python.stdout.on("data", (data) => {
    output += data.toString();
  });

  python.stderr.on("data", (data) => {
    // errorOutput += data.toString();
    // console.error("PYTHON ERROR:", data.toString());
  });

  python.on("close", () => {
    if (errorOutput.trim() !== "") {
      return res.status(500).json({
        error: "Python execution error",
        details: errorOutput,
      });
    }

    try {
      const json = JSON.parse(output);
      res.json(json);
    } catch (err) {
      res.status(500).json({
        error: "Invalid JSON from Python",
        output,
      });
    }
  });
});

app.listen(3000, () =>
  console.log("🚀 Express server running on http://localhost:3000")
);
