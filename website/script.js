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
