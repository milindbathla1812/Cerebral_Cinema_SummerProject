// ============================
// Cerebral Cinema
// script.js
// ============================

// Smooth fade-in animations

const observer = new IntersectionObserver((entries)=>{

    entries.forEach(entry=>{

        if(entry.isIntersecting){

            entry.target.classList.add("show");

        }

    });

},{
    threshold:0.15
});

document.querySelectorAll("section,.card").forEach(el=>{

    el.classList.add("hidden");

    observer.observe(el);

});


// ============================
// Navbar Background
// ============================

window.addEventListener("scroll",()=>{

    const nav=document.querySelector("nav");

    if(window.scrollY>80){

        nav.style.background="rgba(5,10,20,.92)";

    }

    else{

        nav.style.background="rgba(8,17,31,.75)";

    }

});


// ============================
// Animated Counters
// ============================

const counters=document.querySelectorAll(".card h2");

let started=false;

window.addEventListener("scroll",()=>{

    if(started) return;

    const stats=document.querySelector(".stats");

    const top=stats.getBoundingClientRect().top;

    if(top<window.innerHeight-100){

        started=true;

        counters.forEach(counter=>{

            const original=counter.innerText;

            const value=parseInt(original);

            if(isNaN(value)) return;

            let current=0;

            const step=Math.ceil(value/80);

            const timer=setInterval(()=>{

                current+=step;

                if(current>=value){

                    counter.innerText=original;

                    clearInterval(timer);

                }

                else{

                    counter.innerText=current;

                }

            },20);

        });

    }

});
// ============================
// Pipeline Interaction
// ============================

document.querySelectorAll(".pipeline-box").forEach(box=>{

    box.addEventListener("click",()=>{

        document.querySelectorAll(".pipeline-box")
        .forEach(b=>b.classList.remove("active"));

        box.classList.add("active");

    });

});

// ============================
// Prediction API
// ============================

const API_URL = "https://cerebral-cinema-summerproject.onrender.com/predict";

const predictBtn = document.getElementById("predictBtn");

if (predictBtn) {

    predictBtn.addEventListener("click", async () => {

        const textFile = document.getElementById("textFile").files[0];
        const videoFile = document.getElementById("videoFile").files[0];
        const fmriFile = document.getElementById("fmriFile").files[0];

        if (!textFile || !videoFile || !fmriFile) {

            alert("Please upload all three .npy files.");
            return;

        }

        const formData = new FormData();

        formData.append("text", textFile);
        formData.append("video", videoFile);
        formData.append("fmri", fmriFile);

        document.getElementById("pearsonScore").innerHTML = "...";
        document.getElementById("statusText").innerHTML = "Running inference...";

        try {

            const response = await fetch(API_URL, {

                method: "POST",
                body: formData

            });

            if (!response.ok) {

                throw new Error("Server Error");

            }

            const result = await response.json();
            
            console.log(result);

            document.getElementById("pearsonScore").innerHTML =
                Number(result.pearson).toFixed(4);

            document.getElementById("statusText").innerHTML =
                "Prediction completed successfully!";

        }

        catch (error) {

            console.error(error);

            document.getElementById("pearsonScore").innerHTML = "--";

            document.getElementById("statusText").innerHTML =
                "Prediction failed.";

            alert("Prediction Failed.");

        }

    });

}