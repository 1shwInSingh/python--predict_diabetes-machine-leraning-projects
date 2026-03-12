async function predict() {
    const data = {
        preg: document.getElementById("preg").value,
        gluc: document.getElementById("gluc").value,
        bp: document.getElementById("bp").value,
        skin: document.getElementById("skin").value,
        ins: document.getElementById("ins").value,
        bmi: document.getElementById("bmi").value,
        dpf: document.getElementById("dpf").value,
        age: document.getElementById("age").value
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    const resultDiv = document.getElementById("result");

    if (result.result === "Diabetic") {
        resultDiv.style.color = "#ff4d4d";
    } else {
        resultDiv.style.color = "#00ff99";
    }

    resultDiv.innerHTML = "Result: " + result.result;
}