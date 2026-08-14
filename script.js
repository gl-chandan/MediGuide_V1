const userBtn = document.querySelector(".user-btn");
// ADMIN MODE

adminBtn.addEventListener("click", function () {

  alert("Admin Mode Activated");

});


// SUBMIT BUTTON

submitBtn.addEventListener("click", function () {

  const symptoms = symptomInput.value.toLowerCase();


  // VALIDATION

  if (symptoms.trim() === "") {

    resultBox.innerText = "Please enter symptoms.";

    resultBox.style.color = "red";

    return;
  }


  // SHOW LOADING

  loading.classList.remove("hidden");

  resultBox.innerText = "";


  // SIMULATE AI PROCESSING

  setTimeout(function () {

    loading.classList.add("hidden");


    // SIMPLE HARD-CODED PREDICTIONS

    if (
      symptoms.includes("fever") &&
      symptoms.includes("cough")
    ) {

      resultBox.innerText =
        "Possible Prediction: Viral Fever or Flu";

    }

    else if (
      symptoms.includes("chest pain")
    ) {

      resultBox.innerText =
        "Possible Prediction: Cardiac Issue";

    }

    else if (
      symptoms.includes("headache")
    ) {

      resultBox.innerText =
        "Possible Prediction: Migraine";

    }

    else {

      resultBox.innerText =
        "Prediction Unclear. Please consult a doctor.";

    }

    resultBox.style.color = "green";

  }, 2000);

});